from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest

from dubber.core.paths import WorkspacePaths
from dubber.providers.base import TTSResult
from dubber.providers.factory import ProviderBundle
from dubber.tts.mock import synthesize_silence_wav, synthesize_tone_wav
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_segment


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
    assert row["alignment_action"] == "time_stretch"
    assert row["stretch_ratio"] == 0.9
    with wave.open(str(paths.root / row["audio_path"]), "rb") as wav:
        duration_ms = int(wav.getnframes() * 1000 / wav.getframerate())
    assert 950 <= duration_ms <= 1050


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

    row = asyncio.run(
        produce_provider_tts_segment(
            paths=paths,
            segment=segment,
            text="Đây là một câu tiếng Việt khá dài cần được rút gọn.",
            provider_bundle=providers,
            ffmpeg=FakeFFmpeg(),
        )
    )

    assert len(llm.calls) == 1
    assert tts.texts == ["Đây là một câu tiếng Việt khá dài cần được rút gọn.", "Ngắn gọn."]
    assert row["tts_duration_ms"] == 1100
    assert row["synthesis_attempts"] == 2
    assert row["rephrase_attempts"] == 1
    assert row["source_text_chars"] == len("Đây là một câu tiếng Việt khá dài cần được rút gọn.")
    assert row["final_text_chars"] == len("Ngắn gọn.")


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
