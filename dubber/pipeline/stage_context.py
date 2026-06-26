from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubber.core.io import read_json
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.providers.factory import ProviderBundle
from dubber.providers.ffmpeg import FFmpegAdapter


@dataclass
class StageContext:
    paths: WorkspacePaths
    store: CheckpointStore
    manifest: ArtifactManifest
    ffmpeg: FFmpegAdapter
    provider_mode: str = "mock"
    provider_bundle: ProviderBundle | None = None

    def artifact_json(self, filename: str) -> dict[str, Any]:
        return read_json(self.paths.artifact_path(filename))

    def resolve_input(self) -> Path:
        return self.paths.resolve_relative(self.store.state.input_file)

    def require_provider_bundle(self) -> ProviderBundle:
        if self.provider_bundle is None:
            raise RuntimeError("provider bundle is not configured")
        return self.provider_bundle
