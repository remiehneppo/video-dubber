from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import pytest

from dubber.core.enums import StageStatus
from dubber.core.models import ASRServiceConfig, DubberConfig, ProjectConfig, RuntimeConfig, SourceNormalizationConfig
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import run_asr
from dubber.providers.base import ASRResult


class FakeFFmpeg:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, int, int]] = []

    def extract_audio_segment(self, input_audio: Path, output_wav: Path, *, start_ms: int, duration_ms: int) -> None:
        self.calls.append((input_audio, output_wav, start_ms, duration_ms))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"RIFFfake")


class FakeASR:
    def __init__(self) -> None:
        self.audio_paths: list[Path] = []

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        self.audio_paths.append(audio_path)
        if audio_path.stem == "seg_000001":
            raw = {
                "text": "First sentence.",
                "language": language,
                "words": [
                    {"word": "First", "start": 0.0, "end": 0.5},
                    {"word": "sentence.", "start": 0.6, "end": 1.2},
                ],
            }
        else:
            raw = {
                "text": "Second sentence.",
                "language": language,
                "words": [
                    {"word": "Second", "start": 0.0, "end": 0.5},
                    {"word": "sentence.", "start": 0.6, "end": 1.2},
                ],
            }
        return ASRResult(text=str(raw["text"]), confidence=0.9, language=language, raw=raw)


class FakeProviderBundle:
    def __init__(self) -> None:
        self.asr = FakeASR()
        self.llm = None
        self.tts = None


class FailsSecondSegmentOnceASR(FakeASR):
    def __init__(self) -> None:
        super().__init__()
        self.call_counts: Counter[str] = Counter()

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        self.call_counts[audio_path.stem] += 1
        if audio_path.stem == "seg_000001":
            await asyncio.sleep(0.02)
        if audio_path.stem == "seg_000002" and self.call_counts[audio_path.stem] == 1:
            raise RuntimeError("temporary ASR failure")
        return await super().transcribe(audio_path, language)


class SegmentOnlyASR(FakeASR):
    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        raw = {
            "text": "Segment only.",
            "language": language,
            "segments": [{"text": "Segment only.", "start": 0.0, "end": 1.0}],
        }
        return ASRResult(text=str(raw["text"]), confidence=0.9, language=language, raw=raw)


class MissingWordsOnceASR(FakeASR):
    def __init__(self) -> None:
        super().__init__()
        self.call_counts: Counter[str] = Counter()

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        self.call_counts[audio_path.stem] += 1
        if audio_path.stem == "seg_000001" and self.call_counts[audio_path.stem] == 1:
            raw = {
                "text": "Segment only.",
                "segments": [{"text": "Segment only.", "start": 0.0, "end": 1.0}],
            }
            return ASRResult(text=str(raw["text"]), confidence=0.9, language=language, raw=raw)
        return await super().transcribe(audio_path, language)


class AmbiguousDoctorASR(FakeASR):
    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        raw = {
            "text": "The doctor appears in the next symbol.",
            "language": language,
            "words": [
                {"word": "The", "start": 0.0, "end": 0.2},
                {"word": "doctor", "start": 0.3, "end": 0.6},
                {"word": "appears", "start": 0.7, "end": 1.0},
                {"word": "in", "start": 1.1, "end": 1.2},
                {"word": "the", "start": 1.3, "end": 1.4},
                {"word": "next", "start": 1.5, "end": 1.7},
                {"word": "symbol.", "start": 1.8, "end": 2.0},
            ],
        }
        return ASRResult(text=str(raw["text"]), confidence=0.9, language=language, raw=raw)


class SourceNormalizationSuggestionLLM:
    def __init__(self) -> None:
        self.prompts: list[dict[str, object]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        self.prompts.append(payload)
        return {
            "suggestions": [
                {
                    "segment_id": payload["candidates"][0]["segment_id"],
                    "candidate_id": payload["candidates"][0]["candidate_id"],
                    "original": "doctor",
                    "suggested_normalized": "dr",
                    "confidence": 0.84,
                    "reason": "May be the calculus differential dr, but user review is required.",
                }
            ]
        }


def test_run_asr_transcribes_per_segment_audio(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr")
    _write_segments_file(paths)
    source_audio = paths.audio_dir / "vocals.wav"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"RIFFfake")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_asr", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_asr", paths.manifest_file)
    ffmpeg = FakeFFmpeg()
    provider_bundle = FakeProviderBundle()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
    )

    run_asr(ctx)

    assert len(ffmpeg.calls) == 2
    assert all(call[0] == source_audio for call in ffmpeg.calls)
    assert sorted(path.name for path in provider_bundle.asr.audio_paths) == ["seg_000001.wav", "seg_000002.wav"]
    assert all(path.parent.name == "asr" for path in provider_bundle.asr.audio_paths)
    transcript = json.loads((paths.artifact_path("transcript.v1.json")).read_text(encoding="utf-8"))
    word_timeline = json.loads((paths.artifact_path("word_timeline.v1.json")).read_text(encoding="utf-8"))
    source_normalization = json.loads((paths.artifact_path("source_normalization.v1.json")).read_text(encoding="utf-8"))
    assert [segment["source_text"] for segment in transcript["segments"]] == ["First sentence.", "Second sentence."]
    assert [segment["source_text_raw"] for segment in transcript["segments"]] == ["First sentence.", "Second sentence."]
    assert source_normalization["segments"][0]["normalization_edits"] == []
    assert transcript["segments"][0]["timestamp_source"] == "word"
    assert transcript["segments"][0]["source_chunk_id"] == "seg_000001"
    assert [word["text"] for word in word_timeline["words"]] == [
        "First",
        "sentence.",
        "Second",
        "sentence.",
    ]
    assert word_timeline["words"][0]["segment_id"] == "seg_000001"
    assert store.state.stages["asr"].status == StageStatus.COMPLETED


def test_run_asr_logs_progress_summary(tmp_path: Path, caplog) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_logs")
    _write_segments_file(paths)
    source_audio = paths.audio_dir / "vocals.wav"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"RIFFfake")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_asr_logs", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_asr_logs", paths.manifest_file)
    ffmpeg = FakeFFmpeg()
    provider_bundle = FakeProviderBundle()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
    )

    caplog.set_level(logging.INFO)
    run_asr(ctx)

    progress_logs = [record.message for record in caplog.records if "stage asr segment completed" in record.message]
    assert any("progress=50.0%" in message for message in progress_logs)
    assert any("progress=100.0%" in message for message in progress_logs)
    assert any("eta_ms=" in message for message in progress_logs)


def test_run_asr_checkpoints_in_flight_success_and_resume_only_retries_failure(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_resume")
    _write_segments_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_resume", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_resume", paths.manifest_file)
    asr = FailsSecondSegmentOnceASR()
    provider_bundle = FakeProviderBundle()
    provider_bundle.asr = asr
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
        config=DubberConfig(runtime=RuntimeConfig(asr_concurrency=2)),
    )

    with pytest.raises(RuntimeError, match="temporary ASR failure"):
        run_asr(ctx)

    progress = json.loads(paths.artifact_path("asr_segments.v1.json").read_text(encoding="utf-8"))
    statuses = {item["segment_id"]: item["status"] for item in progress["segments"]}
    assert statuses == {"seg_000001": "completed", "seg_000002": "failed"}

    run_asr(ctx)

    assert asr.call_counts == Counter({"seg_000002": 2, "seg_000001": 1})
    transcript = json.loads(paths.artifact_path("transcript.v1.json").read_text(encoding="utf-8"))
    assert [segment["source_text"] for segment in transcript["segments"]] == [
        "First sentence.",
        "Second sentence.",
    ]


def test_run_asr_requires_word_level_timestamps_in_production(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_word_required")
    _write_segments_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_word_required", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_word_required", paths.manifest_file)
    provider_bundle = FakeProviderBundle()
    provider_bundle.asr = SegmentOnlyASR()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
        config=DubberConfig(asr_service=ASRServiceConfig(require_word_timestamps=True)),
    )

    with pytest.raises(ValueError, match="word-level timestamps"):
        run_asr(ctx)


def test_run_asr_resume_retries_unit_with_invalid_timestamp_payload(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_invalid_timestamp_resume")
    _write_segments_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_asr_invalid_timestamp_resume",
        input_file=Path("input/video.mp4"),
    )
    manifest = ArtifactManifest.create("job_asr_invalid_timestamp_resume", paths.manifest_file)
    asr = MissingWordsOnceASR()
    provider_bundle = FakeProviderBundle()
    provider_bundle.asr = asr
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
        config=DubberConfig(runtime=RuntimeConfig(asr_concurrency=2)),
    )

    with pytest.raises(ValueError, match="word-level timestamps"):
        run_asr(ctx)

    progress = json.loads(paths.artifact_path("asr_segments.v1.json").read_text(encoding="utf-8"))
    statuses = {item["segment_id"]: item["status"] for item in progress["segments"]}
    assert statuses["seg_000001"] == "failed"

    legacy_checkpoint = SegmentCheckpointStore.load(paths.artifact_path("asr_segments.v1.json"))
    legacy_checkpoint.mark(
        "seg_000001",
        StageStatus.COMPLETED,
        artifact="raw/asr/seg_000001.json",
    )
    legacy_checkpoint.save()

    run_asr(ctx)

    assert asr.call_counts["seg_000001"] == 2
    assert asr.call_counts["seg_000002"] == 1


def test_run_asr_rejects_disabling_word_timestamps_in_provider_mode(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_word_disabled")
    _write_segments_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_word_disabled", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_word_disabled", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=FakeProviderBundle(),  # type: ignore[arg-type]
        config=DubberConfig(asr_service=ASRServiceConfig(require_word_timestamps=False)),
    )

    with pytest.raises(ValueError, match="production ASR requires word-level timestamps"):
        run_asr(ctx)


def test_run_asr_preserves_vad_risk_provenance_in_transcript(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_vad_risk")
    _write_segments_file(paths)
    segments_path = paths.artifact_path("segments.v1.json")
    segments_payload = json.loads(segments_path.read_text(encoding="utf-8"))
    segments_payload["segments"][0].update({
        "split_reason": "vad_hard_split",
        "risk_flags": ["hard_split"],
        "silence_before_ms": 0,
        "silence_after_ms": 0,
    })
    segments_path.write_text(json.dumps(segments_payload), encoding="utf-8")
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_vad_risk", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_vad_risk", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=FakeProviderBundle(),  # type: ignore[arg-type]
    )

    run_asr(ctx)

    transcript = json.loads(paths.artifact_path("transcript.v1.json").read_text(encoding="utf-8"))
    first = transcript["segments"][0]
    assert first["vad_split_reason"] == "vad_hard_split"
    assert first["vad_risk_flags"] == ["hard_split"]
    assert "hard_split" in first["risk_flags"]


def test_run_asr_records_llm_source_normalization_suggestions_without_applying_them(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_source_suggestions")
    _write_single_segment_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_source_suggestions", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_source_suggestions", paths.manifest_file)
    provider_bundle = FakeProviderBundle()
    provider_bundle.asr = AmbiguousDoctorASR()
    provider_bundle.llm = SourceNormalizationSuggestionLLM()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            source_normalization=SourceNormalizationConfig(llm_adjudication=True),
        ),
    )

    run_asr(ctx)

    transcript = json.loads(paths.artifact_path("transcript.v1.json").read_text(encoding="utf-8"))
    segment = transcript["segments"][0]
    assert segment["source_text"] == "The doctor appears in the next symbol."
    assert segment["normalization_edits"] == []
    assert segment["normalization_suggestions"][0]["suggested_normalized"] == "dr"
    assert "source_normalization_llm_suggestion" in segment["risk_flags"]
    source_normalization = json.loads(paths.artifact_path("source_normalization.v1.json").read_text(encoding="utf-8"))
    assert source_normalization["segments"][0]["normalization_suggestions"][0]["original"] == "doctor"
    assert provider_bundle.llm.prompts[0]["domain_profile"] == "calculus@1"


def _write_segments_file(paths: WorkspacePaths) -> None:
    paths.artifact_path("segments.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": paths.root.name,
                "source_audio": "audio/vocals.wav",
                "segments": [
                    {"segment_id": "seg_000001", "start_ms": 0, "end_ms": 1500, "duration_ms": 1500},
                    {"segment_id": "seg_000002", "start_ms": 1500, "end_ms": 3200, "duration_ms": 1700},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_single_segment_file(paths: WorkspacePaths) -> None:
    paths.artifact_path("segments.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": paths.root.name,
                "source_audio": "audio/vocals.wav",
                "segments": [
                    {"segment_id": "seg_000001", "start_ms": 0, "end_ms": 2500, "duration_ms": 2500},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
