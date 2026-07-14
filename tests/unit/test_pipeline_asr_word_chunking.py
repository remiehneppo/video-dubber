from __future__ import annotations

import json
from pathlib import Path

from dubber.core.enums import StageStatus
from dubber.core.models import ASRChunkingConfig, DubberConfig
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


class GapWordASR:
    def __init__(self) -> None:
        self.audio_paths: list[Path] = []

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        self.audio_paths.append(audio_path)
        raw = {
            "text": "First sentence. Later phrase.",
            "language": language,
            "words": [
                {"word": "First", "start": 0.0, "end": 0.5},
                {"word": "sentence.", "start": 0.6, "end": 1.2},
                {"word": "Later", "start": 6.4, "end": 6.9},
                {"word": "phrase.", "start": 7.0, "end": 7.6},
            ],
        }
        return ASRResult(text=str(raw["text"]), confidence=0.9, language=language, raw=raw)


class FakeProviderBundle:
    def __init__(self) -> None:
        self.asr = GapWordASR()
        self.llm = None
        self.tts = None


def test_run_asr_word_chunking_emits_artifact_and_transcript_source_chunks(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_word_chunks")
    _write_single_long_segment_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_word_chunks", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_word_chunks", paths.manifest_file)
    provider_bundle = FakeProviderBundle()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
        config=DubberConfig(asr_chunking=ASRChunkingConfig(enabled=True)),
    )

    run_asr(ctx)

    word_chunks = json.loads(paths.artifact_path("asr_word_chunks.v1.json").read_text(encoding="utf-8"))
    transcript = json.loads(paths.artifact_path("transcript.v1.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(paths.artifact_path("asr_segments.v1.json").read_text(encoding="utf-8"))
    assert [chunk["chunk_id"] for chunk in word_chunks["chunks"]] == ["wchunk_000001", "wchunk_000002"]
    assert [segment["source_chunk_id"] for segment in transcript["segments"]] == [
        "wchunk_000001",
        "wchunk_000002",
    ]
    assert [segment["source_text"] for segment in transcript["segments"]] == [
        "First sentence.",
        "Later phrase.",
    ]
    assert [item["segment_id"] for item in checkpoint["segments"]] == ["seg_000001"]
    assert word_chunks["chunks"][0]["source_bootstrap_segment_ids"] == ["seg_000001"]


def test_run_asr_word_chunking_resume_rebuilds_without_asr_call(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_asr_word_chunks_resume")
    _write_single_long_segment_file(paths)
    (paths.audio_dir / "vocals.wav").write_bytes(b"RIFFfake")
    raw_path = paths.raw_dir / "asr" / "seg_000001.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "text": "First sentence. Later phrase.",
                "confidence": 0.8,
                "words": [
                    {"word": "First", "start": 0.0, "end": 0.5},
                    {"word": "sentence.", "start": 0.6, "end": 1.2},
                    {"word": "Later", "start": 6.4, "end": 6.9},
                    {"word": "phrase.", "start": 7.0, "end": 7.6},
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = SegmentCheckpointStore.load_or_create(
        paths.artifact_path("asr_segments.v1.json"),
        stage="asr",
        segment_ids=["seg_000001"],
    )
    checkpoint.mark("seg_000001", StageStatus.COMPLETED, artifact="raw/asr/seg_000001.json")
    checkpoint.save()
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_asr_word_chunks_resume", input_file=Path("input/video.mp4")
    )
    manifest = ArtifactManifest.create("job_asr_word_chunks_resume", paths.manifest_file)
    provider_bundle = FakeProviderBundle()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=provider_bundle,  # type: ignore[arg-type]
        config=DubberConfig(asr_chunking=ASRChunkingConfig(enabled=True)),
    )

    run_asr(ctx)

    assert provider_bundle.asr.audio_paths == []
    assert paths.artifact_path("asr_word_chunks.v1.json").exists()
    transcript = json.loads(paths.artifact_path("transcript.v1.json").read_text(encoding="utf-8"))
    assert [segment["source_chunk_id"] for segment in transcript["segments"]] == [
        "wchunk_000001",
        "wchunk_000002",
    ]


def _write_single_long_segment_file(paths: WorkspacePaths) -> None:
    paths.artifact_path("segments.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": paths.root.name,
                "source_audio": "audio/vocals.wav",
                "segments": [
                    {"segment_id": "seg_000001", "start_ms": 0, "end_ms": 12000, "duration_ms": 12000},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
