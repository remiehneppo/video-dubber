from __future__ import annotations

import asyncio
import json
import wave
from collections import Counter
from pathlib import Path

import pytest

from dubber.core.paths import WorkspacePaths
from dubber.providers.base import TTSResult
from dubber.providers.factory import ProviderBundle
from dubber.tts.mock import synthesize_silence_wav, synthesize_tone_wav
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_rows, produce_provider_tts_segment


def test_produce_mock_tts_segment_returns_manifest_row(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }

    row = produce_mock_tts_segment(paths=paths, segment=segment)

    assert row["segment_id"] == "seg_000001"
    assert row["orig_duration_ms"] == 1000
    assert row["tts_duration_ms"] == 1100
    assert row["alignment_action"] == "time_stretch"
    assert row["raw_audio_path"] == "tts/seg_000001.raw.wav"
    assert row["audio_path"] == "tts/seg_000001.wav"
    assert (paths.root / row["audio_path"]).exists()
    with wave.open(str(paths.root / row["audio_path"]), "rb") as wav:
        assert wav.getnchannels() == 1


class FakeProviderTTS:
    def __init__(self) -> None:
        self.voice = "Trúc Ly"
        self.seen_voice = ""
        self.texts: list[str] = []

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        self.seen_voice = voice
        self.texts.append(text)
        synthesize_tone_wav(output_path, 900)
        return TTSResult(audio_path=output_path, duration_ms=900, provider_metadata={"voice": voice})


class SequenceProviderTTS:
    def __init__(self, durations_ms: list[int], *, silent_first: bool = False) -> None:
        self.voice = "Trúc Ly"
        self.durations_ms = durations_ms
        self.silent_first = silent_first
        self.texts: list[str] = []

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        self.texts.append(text)
        duration_ms = self.durations_ms[min(len(self.texts) - 1, len(self.durations_ms) - 1)]
        if self.silent_first and len(self.texts) == 1:
            synthesize_silence_wav(output_path, duration_ms)
        else:
            synthesize_tone_wav(output_path, duration_ms)
        return TTSResult(
            audio_path=output_path,
            duration_ms=duration_ms,
            provider_metadata={"attempt": len(self.texts)},
        )


class EdgeSilenceProviderTTS:
    voice = "Trúc Ly"

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        _write_samples(output_path, [0] * 1500 + [3000] * 700, sample_rate=1000)
        return TTSResult(audio_path=output_path, duration_ms=2200, provider_metadata={"voice": voice})


class FakeLLM:
    def __init__(self, response_text: str = "Bản ngắn hơn.") -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    async def complete_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, object],
        *,
        response_name: str,
        response_description: str,
    ) -> dict[str, object]:
        assert "shorten Vietnamese dubbing text" in system_prompt
        assert "Preserve the original meaning" in system_prompt
        assert "formulas, symbols, numbers, units" in system_prompt
        assert "remove redundancy, not information" in system_prompt
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": schema,
                "response_name": response_name,
                "response_description": response_description,
            }
        )
        return {"text": self.response_text}


class FakeFFmpeg:
    def duration_ms(self, audio_path: Path) -> int:
        return 900


class SequenceASR:
    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = transcripts
        self.languages: list[str] = []

    async def transcribe(self, audio_path: Path, language: str):
        from dubber.providers.base import ASRResult

        self.languages.append(language)
        text = self.transcripts[min(len(self.languages) - 1, len(self.transcripts) - 1)]
        return ASRResult(text=text, confidence=1.0, language=language, raw={"text": text})


class RecordingConcurrency:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    async def run_asr(self, operation):
        self.calls["asr"] += 1
        return await operation()

    async def run_llm(self, operation):
        self.calls["llm"] += 1
        return await operation()

    async def run_tts(self, operation):
        self.calls["tts"] += 1
        return await operation()


def _write_samples(path: Path, samples: list[int], *, sample_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


def test_produce_provider_tts_rows_preserves_source_pause_between_clauses(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 4500,
        "duration_ms": 3500,
        "source_text": "First sentence. Second sentence.",
        "timestamp_source": "word",
        "words": [
            {"text": "First", "start_ms": 1000, "end_ms": 1400},
            {"text": "sentence.", "start_ms": 1500, "end_ms": 2000},
            {"text": "Second", "start_ms": 3200, "end_ms": 3600},
            {"text": "sentence.", "start_ms": 3700, "end_ms": 4500},
        ],
    }
    tts = SequenceProviderTTS([1800, 1000])
    llm = FakeLLM()
    providers = ProviderBundle(asr=None, llm=llm, tts=tts)

    rows = asyncio.run(
        produce_provider_tts_rows(
            paths=paths,
            segment=segment,
            text="Cau thu nhat. Cau thu hai.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            next_segment_start_ms=6000,
            clause_pause_threshold_ms=700,
        )
    )

    assert tts.texts == ["Cau thu nhat.", "Cau thu hai."]
    assert [row["segment_id"] for row in rows] == [
        "seg_000001__clause_001",
        "seg_000001__clause_002",
    ]
    assert [row["target_start_ms"] for row in rows] == [1000, 3200]
    assert rows[0]["parent_segment_id"] == "seg_000001"
    assert rows[0]["clause_count"] == 2
    assert rows[0]["alignment_action"] == "overflow"
    assert rows[0]["overflow_ms"] == 800
    assert llm.calls == []
    assert rows[0]["target_start_ms"] + rows[0]["orig_duration_ms"] == 2000
    assert rows[1]["target_start_ms"] - 2000 == 1200


def test_produce_provider_tts_segment_retries_semantic_mismatch_in_vietnamese(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_semantic")
    segment = {"segment_id": "cue_1", "start_ms": 0, "end_ms": 1000, "duration_ms": 1000}
    tts = SequenceProviderTTS([900, 900])
    asr = SequenceASR(["đọc hoàn toàn sai", "xin chào"])
    providers = ProviderBundle(asr=asr, llm=FakeLLM(), tts=tts)
    concurrency = RecordingConcurrency()

    row = asyncio.run(produce_provider_tts_segment(
        paths=paths,
        segment=segment,
        text="Xin chào",
        provider_bundle=providers,
        ffmpeg=FakeFFmpeg(),
        semantic_retry_attempts=3,
        concurrency=concurrency,  # type: ignore[arg-type]
    ))

    assert len(tts.texts) == 2
    assert asr.languages == ["vi", "vi"]
    assert row["final_text"] == "Xin chào"
    assert row["semantic_metrics"]["cer"] == 0.0
    assert row["quality_attempts"][0]["semantic_ok"] is False
    assert concurrency.calls == Counter({"tts": 2, "asr": 2})


def test_semantic_failure_preserves_attempt_trace_for_resume(tmp_path: Path) -> None:
    import json

    paths = WorkspacePaths.create(tmp_path, "job_semantic_failure")
    segment = {"segment_id": "cue_bad", "start_ms": 0, "end_ms": 1000, "duration_ms": 1000}
    providers = ProviderBundle(
        asr=SequenceASR(["sai hoàn toàn"]),
        llm=FakeLLM(),
        tts=SequenceProviderTTS([900]),
    )

    with pytest.raises(ValueError, match="tts_semantic_quality_failed"):
        asyncio.run(produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Xin chào",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            semantic_retry_attempts=2,
        ))

    trace = json.loads((paths.raw_dir / "tts" / "cue_bad.quality.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert len([attempt for attempt in trace["quality_attempts"] if attempt.get("semantic_ok") is False]) == 2
    assert "tts_semantic_quality_failed" in trace["final_error"]


def test_produce_provider_tts_segment_uses_provider_voice(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = FakeProviderTTS()
    providers = ProviderBundle(asr=None, llm=None, tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Xin chào",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
        )
    )

    assert tts.seen_voice == "Trúc Ly"
    assert row["provider_metadata"] == {"voice": "Trúc Ly"}
    assert row["alignment_action"] == "pad_silence"
    assert row["stretch_ratio"] == 1.0
    with wave.open(str(paths.root / row["audio_path"]), "rb") as wav:
        duration_ms = int(wav.getnframes() * 1000 / wav.getframerate())
    assert 850 <= duration_ms <= 950


def test_produce_provider_tts_segment_with_empty_text_generates_silence(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = FakeProviderTTS()
    providers = ProviderBundle(asr=None, llm=None, tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="   ",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
        )
    )

    assert tts.seen_voice == ""
    assert row["provider_metadata"]["generated_silence"] is True
    assert row["tts_duration_ms"] == 1000
    assert row["alignment_action"] == "pad_silence"
    assert (paths.root / row["audio_path"]).exists()



def test_produce_provider_tts_segment_retries_near_silent_audio(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = SequenceProviderTTS([900, 900], silent_first=True)
    providers = ProviderBundle(asr=None, llm=FakeLLM(), tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Xin chào",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
        )
    )

    assert len(tts.texts) == 2
    assert row["synthesis_attempts"] == 2
    assert row["rephrase_attempts"] == 0
    assert row["quality_report"]["ok"] is True
    assert "tts_audio_near_silent" in row["quality_attempts"][0]["warnings"]


def test_produce_provider_tts_segment_trims_leading_silence(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 3000,
        "duration_ms": 2000,
    }
    providers = ProviderBundle(asr=None, llm=FakeLLM(), tts=EdgeSilenceProviderTTS())

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Xin chào",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            max_edge_silence_ms=1200,
        )
    )

    assert row["provider_metadata"]["edge_silence_trimmed"] is True
    assert "tts_audio_leading_silence" in row["quality_attempts"][0]["warnings"]
    assert row["quality_report"]["leading_silence_ms"] == 100
    assert row["quality_report"]["ok"] is True


def test_produce_provider_tts_segment_rephrases_audio_that_exceeds_max_speedup(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = SequenceProviderTTS([1700, 1100])
    llm = FakeLLM("Ngắn gọn.")
    providers = ProviderBundle(asr=None, llm=llm, tts=tts)
    concurrency = RecordingConcurrency()

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Đây là một câu tiếng Việt khá dài cần được rút gọn.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            concurrency=concurrency,  # type: ignore[arg-type]
        )
    )

    assert len(llm.calls) == 1
    assert tts.texts == ["Đây là một câu tiếng Việt khá dài cần được rút gọn.", "Ngắn gọn."]
    assert row["tts_duration_ms"] == 1100
    assert row["synthesis_attempts"] == 2
    assert row["rephrase_attempts"] == 1
    assert row["source_text_chars"] == len("Đây là một câu tiếng Việt khá dài cần được rút gọn.")
    assert row["final_text_chars"] == len("Ngắn gọn.")
    assert concurrency.calls == Counter({"tts": 2, "llm": 1})


def test_tts_rephrase_targets_cue_plus_borrowable_silence_window(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_rephrase_overflow_window")
    segment = {
        "segment_id": "cue_overflow_window",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    llm = FakeLLM("Nội dung vừa khung.")
    providers = ProviderBundle(
        asr=None,
        llm=llm,
        tts=SequenceProviderTTS([2000, 1000]),
    )

    asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Nội dung kỹ thuật ban đầu dài hơn cửa sổ cho phép.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            next_segment_start_ms=2300,
            max_overflow_ms=300,
            overflow_reserve_ms=0,
            max_speedup_ratio=1.2,
        )
    )

    prompt = json.loads(str(llm.calls[0]["user_prompt"]))
    assert prompt["target_duration_ms"] == 1300


def test_tts_rephrase_rejects_protected_calculus_notation_change(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_rephrase_protected_span")
    segment = {
        "segment_id": "cue_protected_dr",
        "start_ms": 0,
        "end_ms": 1000,
        "duration_ms": 1000,
        "source_text": "The area includes dr.",
        "protected_spans": [
            {
                "canonical": "dr",
                "spoken": "d r",
                "display": "dr",
                "forbidden": ["doctor", "bác sĩ", "tiến sĩ"],
            }
        ],
    }
    tts = SequenceProviderTTS([1800])
    providers = ProviderBundle(asr=None, llm=FakeLLM("Diện tích nhân doctor."), tts=tts)

    with pytest.raises(ValueError, match="tts_rephrase_protected_span_violation"):
        asyncio.run(
            produce_provider_tts_segment(
                paths=paths,
                segment=segment,
                text="Diện tích nhân d r.",
                provider_bundle=providers,
                ffmpeg=FakeFFmpeg(),
                max_speedup_ratio=1.2,
            )
        )

    assert len(tts.texts) == 1


def test_rephrase_gets_a_fresh_semantic_retry_budget(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_rephrase_semantic_budget")
    segment = {
        "segment_id": "cue_retry_budget",
        "start_ms": 0,
        "end_ms": 1000,
        "duration_ms": 1000,
    }
    original = "Nội dung ban đầu"
    shortened = "Nội dung ngắn"
    providers = ProviderBundle(
        asr=SequenceASR(["sai", "vẫn sai", original, "sai", shortened]),
        llm=FakeLLM(shortened),
        tts=SequenceProviderTTS([1700, 1700, 1700, 900, 900]),
    )

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text=original,
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            semantic_retry_attempts=3,
        )
    )

    assert row["final_text"] == shortened
    assert row["semantic_metrics"]["token_recall"] == 1.0
    assert row["synthesis_attempts"] == 5
    assert row["rephrase_attempts"] == 1


def test_produce_provider_tts_segment_uses_silence_and_mild_speedup_before_rephrase(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = SequenceProviderTTS([1700, 1500])
    providers = ProviderBundle(asr=None, llm=FakeLLM("Vẫn vừa khe trống."), tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Nội dung dài.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            rephrase_attempts=1,
            next_segment_start_ms=3000,
        )
    )

    assert row["alignment_action"] == "overflow"
    assert row["stretch_ratio"] == 1.0
    assert row["overflow_ms"] == 700
    assert row["target_start_ms"] == 1000
    assert row["target_end_ms"] == 2700
    assert row["rephrase_attempts"] == 0
    assert providers.llm.calls == []


def test_produce_provider_tts_segment_respects_overflow_cap_and_reserve(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_overflow_cap")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = SequenceProviderTTS([1400])
    providers = ProviderBundle(asr=None, llm=FakeLLM("Không được gọi."), tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Nội dung vừa đủ dài.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            next_segment_start_ms=3000,
            max_overflow_ms=250,
            overflow_reserve_ms=200,
            max_speedup_ratio=1.3,
        )
    )

    assert row["available_overflow_ms"] == 250
    assert row["target_window_ms"] == 1250
    assert row["alignment_action"] == "time_stretch_overflow"
    assert row["overflow_ms"] == 250
    assert row["target_end_ms"] == 2250
    assert row["stretch_ratio"] == 1.12
    assert providers.llm.calls == []


def test_produce_provider_tts_segment_compacts_excessive_internal_silence(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = FakeProviderTTS()

    async def synthesize_with_long_pauses(text: str, voice: str, output_path: Path) -> TTSResult:
        tts.texts.append(text)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(1000)
            samples = [3000] * 200 + [180] * 3000 + [3000] * 200
            wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
        return TTSResult(audio_path=output_path, duration_ms=3400, provider_metadata={})

    tts.synthesize = synthesize_with_long_pauses  # type: ignore[method-assign]
    providers = ProviderBundle(asr=None, llm=FakeLLM(), tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Text with an abnormally long pause.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
        )
    )

    assert len(tts.texts) == 1
    assert row["tts_duration_ms"] == 900
    assert row["provider_metadata"]["internal_silence_compacted"] is True
    assert row["provider_metadata"]["duration_before_compaction_ms"] == 3400
    assert row["provider_metadata"]["duration_after_compaction_ms"] == 900
    assert len(row["quality_attempts"]) == 2
    assert "tts_audio_internal_silence" in row["quality_attempts"][0]["warnings"]
    assert row["quality_report"]["ok"] is True


def test_produce_provider_tts_segment_fails_when_rephrased_audio_is_still_too_long(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = SequenceProviderTTS([1700, 1600, 1500])
    providers = ProviderBundle(asr=None, llm=FakeLLM("Vẫn dài."), tts=tts)

    with pytest.raises(ValueError, match="seg_000001.*tts_duration_exceeds_max_speedup"):
        asyncio.run(
            produce_provider_tts_segment(
                paths=paths,
                segment=segment,
                text="Nội dung dài.",
                provider_bundle=providers,
                ffmpeg=FakeFFmpeg(),
                rephrase_attempts=2,
            )
        )


def test_produce_provider_tts_segment_fails_after_repeated_bad_quality_audio(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }
    tts = SequenceProviderTTS([900, 900, 900], silent_first=True)

    async def always_silent(text: str, voice: str, output_path: Path) -> TTSResult:
        tts.texts.append(text)
        synthesize_silence_wav(output_path, 900)
        return TTSResult(audio_path=output_path, duration_ms=900, provider_metadata={})

    tts.synthesize = always_silent  # type: ignore[method-assign]
    providers = ProviderBundle(asr=None, llm=FakeLLM(), tts=tts)

    with pytest.raises(ValueError, match="seg_000001.*tts_audio_near_silent"):
        asyncio.run(
            produce_provider_tts_segment(
                paths=paths,
                segment=segment,
                text="Xin chào",
                provider_bundle=providers,
                ffmpeg=FakeFFmpeg(),
                quality_retry_attempts=3,
            )
        )



def test_provider_tts_uses_trailing_silence_then_only_required_speedup(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_trailing_silence")
    segment = {
        "segment_id": "seg_000010",
        "start_ms": 1000,
        "end_ms": 4220,
        "duration_ms": 3220,
    }
    tts = SequenceProviderTTS([7420])
    llm = FakeLLM("Không được gọi.")
    providers = ProviderBundle(asr=None, llm=llm, tts=tts)

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Về cuối, xe chậm lại nên đường cong thoải dần.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
            next_segment_start_ms=8560,
            max_speedup_ratio=1.3,
        )
    )

    assert llm.calls == []
    assert tts.texts == ["Về cuối, xe chậm lại nên đường cong thoải dần."]
    assert row["alignment_action"] == "overflow"
    assert row["stretch_ratio"] == 1.0
    assert row["available_overflow_ms"] == 4220
    assert row["target_window_ms"] == 7440
    assert row["overflow_ms"] == 4200
    assert row["target_end_ms"] == 8420
    with wave.open(str(paths.root / row["audio_path"]), "rb") as wav:
        aligned_duration_ms = int(wav.getnframes() * 1000 / wav.getframerate())
    assert 7400 <= aligned_duration_ms <= 7440
