from __future__ import annotations

import json
from pathlib import Path

import pytest

from dubber.core.config import load_config
from dubber.core.enums import JobStatus, StageName, StageStatus
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore


def test_load_config_applies_defaults_and_env_placeholders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "project:",
                "  workspace_dir: ${WORKSPACE_DIR}",
                "  output_dir: ./out",
                "runtime:",
                "  asr_concurrency: 3",
                "mixing:",
                "  original_ducking_db: -18",
                "  tts_boost_db: 8.0",
                "  final_loudness_normalization: false",
                "subtitles:",
                "  enabled: true",
                "  mode: burn_in",
                "  output_sidecar: false",
                "  max_height_ratio: 0.12",
                "  background_opacity: 0.5",
                "  font_family: Noto Sans",
                "  font_size_ratio: 0.03",
                "  bottom_margin_ratio: 0.04",
                "  source_enabled: true",
                "  translation_enabled: false",
                "  max_cue_duration_ms: 3000",
                "  min_cue_duration_ms: 600",
                "  max_chars_per_line: 42",
                "vad:",
                "  frame_ms: 50",
                "  threshold_ratio: 0.2",
                "  min_speech_duration_ms: 150",
                "  target_min_chunk_ms: 500",
                "  preferred_max_chunk_ms: 900",
                "  hard_max_chunk_ms: 1200",
                "  silence_merge_threshold_ms: 250",
                "  context_padding_ms: 300",
                "  soft_split_allowed: false",
                "translation:",
                "  glossary_review: false",
                "dubbing_cues:",
                "  target_duration_ms: 3800",
                "  min_duration_ms: 1400",
                "  max_duration_ms: 5900",
                "tts_service:",
                "  quality_retry_attempts: 4",
                "  rephrase_attempts: 1",
                "  max_speedup_ratio: 1.25",
                "  min_rms: 450",
                "  silence_rms_threshold: 100",
                "  max_edge_silence_ms: 1500",
                "  max_internal_silence_ms: 2000",
                "  clipping_peak_threshold: 32700",
                "  max_clipped_sample_ratio: 0.002",
                "  clause_pause_threshold_ms: 850",
                "  max_overflow_ms: 4200",
                "  overflow_reserve_ms: 180",
                "  start_delay_ms: 25",
                "  retained_edge_silence_ms: 90",
                "  semantic_max_cer: 0.2",
                "  semantic_min_token_recall: 0.9",
                "  semantic_retry_attempts: 5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "jobs"))

    config = load_config(config_file)

    assert config.project.workspace_dir == tmp_path / "jobs"
    assert config.project.output_dir == Path("out")
    assert config.runtime.asr_concurrency == 3
    assert config.runtime.tts_concurrency == 4
    assert config.mixing.original_ducking_db == -18.0
    assert config.mixing.tts_boost_db == 8.0
    assert config.mixing.final_loudness_normalization is False
    assert config.subtitles.enabled is True
    assert config.subtitles.mode == "burn_in"
    assert config.subtitles.output_sidecar is False
    assert config.subtitles.max_height_ratio == 0.12
    assert config.subtitles.background_opacity == 0.5
    assert config.subtitles.font_family == "Noto Sans"
    assert config.subtitles.font_size_ratio == 0.03
    assert config.subtitles.bottom_margin_ratio == 0.04
    assert config.subtitles.source_enabled is True
    assert config.subtitles.translation_enabled is False
    assert config.subtitles.max_cue_duration_ms == 3000
    assert config.subtitles.min_cue_duration_ms == 600
    assert config.subtitles.max_chars_per_line == 42
    assert config.vad.frame_ms == 50
    assert config.vad.threshold_ratio == 0.2
    assert config.vad.min_speech_duration_ms == 150
    assert config.vad.target_min_chunk_ms == 500
    assert config.vad.preferred_max_chunk_ms == 900
    assert config.vad.hard_max_chunk_ms == 1200
    assert config.vad.silence_merge_threshold_ms == 250
    assert config.vad.context_padding_ms == 300
    assert config.vad.soft_split_allowed is False
    assert config.translation.glossary_review is False
    assert config.dubbing_cues.target_duration_ms == 3800
    assert config.dubbing_cues.min_duration_ms == 1400
    assert config.dubbing_cues.max_duration_ms == 5900
    assert config.tts_service.quality_retry_attempts == 4
    assert config.tts_service.rephrase_attempts == 1
    assert config.tts_service.max_speedup_ratio == 1.25
    assert config.tts_service.min_rms == 450
    assert config.tts_service.silence_rms_threshold == 100
    assert config.tts_service.max_edge_silence_ms == 1500
    assert config.tts_service.max_internal_silence_ms == 2000
    assert config.tts_service.clipping_peak_threshold == 32700
    assert config.tts_service.max_clipped_sample_ratio == 0.002
    assert config.tts_service.clause_pause_threshold_ms == 850
    assert config.tts_service.max_overflow_ms == 4200
    assert config.tts_service.overflow_reserve_ms == 180
    assert config.tts_service.start_delay_ms == 25
    assert config.tts_service.retained_edge_silence_ms == 90
    assert config.tts_service.semantic_max_cer == 0.2
    assert config.tts_service.semantic_min_token_recall == 0.9
    assert config.tts_service.semantic_retry_attempts == 5
    assert config.input.allowed_extensions == [".mp4", ".mkv", ".mov"]


def test_load_config_applies_asr_driven_segmentation_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "vad:",
                "  mode: asr_context_chunks",
                "  min_speech_duration_ms: 700",
                "  target_min_chunk_ms: 20000",
                "  preferred_max_chunk_ms: 45000",
                "  hard_max_chunk_ms: 90000",
                "asr_service:",
                "  timestamp_mode: prefer_word",
                "  require_timestamps: true",
                "  require_word_timestamps: true",
                "  allow_chunk_text_fallback: false",
                "  vad_filter: false",
                "transcript_segmentation:",
                "  target_min_segment_ms: 8000",
                "  preferred_max_segment_ms: 25000",
                "  max_segment_ms: 45000",
                "  min_pause_split_ms: 600",
                "  prefer_punctuation_split: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.vad.mode == "asr_context_chunks"
    assert config.vad.min_speech_duration_ms == 700
    assert config.vad.target_min_chunk_ms == 20_000
    assert config.vad.preferred_max_chunk_ms == 45_000
    assert config.vad.hard_max_chunk_ms == 90_000
    assert config.vad.soft_split_allowed is False
    assert config.asr_service.timestamp_mode == "prefer_word"
    assert config.asr_service.require_timestamps is True
    assert config.asr_service.require_word_timestamps is True
    assert config.asr_service.allow_chunk_text_fallback is False
    assert config.asr_service.vad_filter is False
    assert config.transcript_segmentation.target_min_segment_ms == 8_000
    assert config.transcript_segmentation.preferred_max_segment_ms == 25_000
    assert config.transcript_segmentation.max_segment_ms == 45_000
    assert config.transcript_segmentation.min_pause_split_ms == 600
    assert config.transcript_segmentation.prefer_punctuation_split is True


def test_load_config_rejects_unsupported_vad_mode(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("vad:\n  mode: energy_segments\n", encoding="utf-8")

    with pytest.raises(ValueError, match="vad.mode must be one of: asr_context_chunks, silero_vad"):
        load_config(config_file)


def test_load_config_does_not_expose_removed_vad_max_chunk_field(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("vad:\n  max_vad_chunk_ms: 25000\n", encoding="utf-8")

    config = load_config(config_file)

    assert not hasattr(config.vad, "max_vad_chunk_ms")


@pytest.mark.parametrize("name,value", [("max_parallel_jobs", 0), ("asr_concurrency", -1), ("llm_concurrency", 1.5), ("tts_concurrency", True)])
def test_load_config_rejects_invalid_concurrency(tmp_path: Path, name: str, value: object) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"runtime:\n  {name}: {str(value).lower()}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"runtime\.{name} must be an integer >= 1"):
        load_config(config_file)


def test_workspace_paths_create_expected_layout_and_reject_traversal(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_123")

    assert paths.root == tmp_path / "job_123"
    assert paths.artifacts_dir.exists()
    assert paths.logs_dir.exists()
    assert paths.output_dir.exists()
    assert paths.artifact_path("segments.v1.json") == paths.artifacts_dir / "segments.v1.json"

    with pytest.raises(ValueError, match="Unsafe relative path"):
        paths.resolve_relative("../outside.json")

    with pytest.raises(ValueError, match="Unsafe relative path"):
        paths.resolve_relative("/tmp/outside.json")


def test_artifact_manifest_records_hash_and_detects_corruption(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_abc")
    artifact = paths.artifact_path("segments.v1.json")
    artifact.write_text('{"segments": []}', encoding="utf-8")

    manifest = ArtifactManifest.create("job_abc", paths.manifest_file)
    entry = manifest.record_artifact(
        name="segments",
        version=1,
        path=artifact,
        created_by_stage=StageName.VAD,
        schema_version="1.0",
    )
    manifest.save()

    reloaded = ArtifactManifest.load(paths.manifest_file)
    assert reloaded.get("segments", 1) == entry
    assert reloaded.validate_artifact("segments", 1) is True

    artifact.write_text('{"segments": ["corrupt"]}', encoding="utf-8")

    assert reloaded.validate_artifact("segments", 1) is False


def test_checkpoint_store_persists_stage_status(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_state")
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_state",
        input_file=Path("input/video.mp4"),
    )

    store.mark_stage(StageName.JOB_INIT, StageStatus.COMPLETED, artifact="input_metadata.v1.json")
    store.mark_stage(StageName.ASR, StageStatus.RUNNING, done=2, total=5)
    store.save()

    raw = json.loads(paths.job_state_file.read_text(encoding="utf-8"))
    assert raw["status"] == JobStatus.RUNNING.value
    assert raw["stages"]["job_init"]["status"] == StageStatus.COMPLETED.value
    assert raw["stages"]["asr"]["done"] == 2

    reloaded = CheckpointStore.load(paths.job_state_file)
    assert reloaded.state.stages[StageName.ASR].total == 5
    assert reloaded.state.stages[StageName.JOB_INIT].artifact == "input_metadata.v1.json"


def test_config_example_loads_provider_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.example/v1")
    monkeypatch.setenv("ASR_API_KEY", "asr-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("TTS_BASE_URL", "https://tts.example/v1")
    monkeypatch.setenv("TTS_API_KEY", "tts-key")

    config = load_config(Path("config.example.yaml"))

    assert config.asr_service.base_url == "https://asr.example/v1"
    assert config.asr_service.api_key == "asr-key"
    assert config.llm_service.model == "gpt-4o-mini"
    assert config.tts_service.voice == "nova"
    assert config.tts_service.quality_retry_attempts == 3
    assert config.tts_service.rephrase_attempts == 2
    assert config.tts_service.max_speedup_ratio == 1.3
    assert config.subtitles.enabled is False
    assert config.subtitles.mode == "burn_in"
    assert config.subtitles.background_opacity == 0.5
