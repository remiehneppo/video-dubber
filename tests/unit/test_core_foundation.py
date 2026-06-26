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
                "vad:",
                "  frame_ms: 50",
                "  threshold_ratio: 0.2",
                "  min_duration_ms: 150",
                "  max_duration_ms: 1200",
                "  silence_merge_threshold_ms: 250",
                "  soft_split_allowed: false",
                "translation:",
                "  glossary_review: false",
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
    assert config.vad.frame_ms == 50
    assert config.vad.threshold_ratio == 0.2
    assert config.vad.min_duration_ms == 150
    assert config.vad.max_duration_ms == 1200
    assert config.vad.silence_merge_threshold_ms == 250
    assert config.vad.soft_split_allowed is False
    assert config.translation.glossary_review is False
    assert config.input.allowed_extensions == [".mp4", ".mkv", ".mov"]


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
