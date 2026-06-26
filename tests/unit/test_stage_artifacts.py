from __future__ import annotations

from pathlib import Path

from dubber.core.enums import StageName, StageStatus
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts


def test_publish_json_records_manifest_and_stage(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_test",
        input_file=Path("input/video.mp4"),
    )
    manifest = ArtifactManifest.create("job_test", paths.manifest_file)
    artifacts = StageArtifacts(paths=paths, store=store, manifest=manifest)

    path = artifacts.publish_json(
        stage=StageName.VAD,
        name="segments",
        filename="segments.v1.json",
        payload={"schema_version": "1.0", "segments": []},
        status=StageStatus.COMPLETED,
        done=0,
        total=0,
    )

    assert path == paths.artifact_path("segments.v1.json")
    assert path.exists()
    assert manifest.validate_artifact("segments", 1) is True
    progress = store.state.stages[StageName.VAD]
    assert progress.status == StageStatus.COMPLETED
    assert progress.artifact == "artifacts/segments.v1.json"
    assert progress.done == 0
    assert progress.total == 0


def test_publish_existing_file_records_manifest_and_stage(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_test",
        input_file=Path("input/video.mp4"),
    )
    manifest = ArtifactManifest.create("job_test", paths.manifest_file)
    artifacts = StageArtifacts(paths=paths, store=store, manifest=manifest)
    final_audio = paths.audio_dir / "final_mix.wav"
    final_audio.write_bytes(b"RIFF")

    artifacts.publish_file(
        stage=StageName.MIXING,
        name="final_audio",
        path=final_audio,
        status=StageStatus.RUNNING,
    )

    assert manifest.validate_artifact("final_audio", 1) is True
    progress = store.state.stages[StageName.MIXING]
    assert progress.status == StageStatus.RUNNING
    assert progress.artifact == "audio/final_mix.wav"
