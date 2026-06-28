from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dubber.core.enums import StageStatus
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
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
    assert [path.name for path in provider_bundle.asr.audio_paths] == ["seg_000001.wav", "seg_000002.wav"]
    assert all(path.parent.name == "asr" for path in provider_bundle.asr.audio_paths)
    transcript = json.loads((paths.artifact_path("transcript.v1.json")).read_text(encoding="utf-8"))
    assert [segment["source_text"] for segment in transcript["segments"]] == ["First sentence.", "Second sentence."]
    assert transcript["segments"][0]["timestamp_source"] == "word"
    assert transcript["segments"][0]["source_chunk_id"] == "seg_000001"
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
