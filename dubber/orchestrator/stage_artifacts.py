from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore


@dataclass
class StageArtifacts:
    paths: WorkspacePaths
    store: CheckpointStore
    manifest: ArtifactManifest

    def publish_json(
        self,
        *,
        stage: StageName,
        name: str,
        filename: str,
        payload: dict[str, Any],
        version: int = 1,
        schema_version: str = "1.0",
        status: StageStatus = StageStatus.COMPLETED,
        done: int | None = None,
        total: int | None = None,
    ) -> Path:
        path = self.paths.artifact_path(filename)
        write_json_atomic(path, payload)
        self.publish_file(
            stage=stage,
            name=name,
            path=path,
            version=version,
            schema_version=schema_version,
            status=status,
            done=done,
            total=total,
        )
        return path

    def publish_file(
        self,
        *,
        stage: StageName,
        name: str,
        path: Path,
        version: int = 1,
        schema_version: str = "1.0",
        status: StageStatus = StageStatus.COMPLETED,
        done: int | None = None,
        total: int | None = None,
    ) -> Path:
        self.manifest.record_artifact(
            name=name,
            version=version,
            path=path,
            created_by_stage=stage,
            schema_version=schema_version,
        )
        self.manifest.save()
        self.store.mark_stage(
            stage,
            status,
            artifact=self.paths.to_relative(path),
            done=done,
            total=total,
        )
        self.store.save()
        return path
