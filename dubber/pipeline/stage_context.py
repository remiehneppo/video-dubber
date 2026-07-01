from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dubber.core.io import read_json
from dubber.core.models import DubberConfig
from dubber.core.concurrency import ProviderConcurrency, validate_threaded_provider_clients
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
    ffmpeg: FFmpegAdapter | None
    config: DubberConfig = field(default_factory=DubberConfig)
    provider_mode: str = "mock"
    provider_bundle: ProviderBundle | None = None
    concurrency: ProviderConcurrency | None = None

    def artifact_json(self, filename: str) -> dict[str, Any]:
        return read_json(self.paths.artifact_path(filename))

    def resolve_input(self) -> Path:
        return self.paths.resolve_relative(self.store.state.input_file)

    def require_provider_bundle(self) -> ProviderBundle:
        if self.provider_bundle is None:
            raise RuntimeError("provider bundle is not configured")
        validate_threaded_provider_clients(self.provider_bundle)
        return self.provider_bundle

    def require_concurrency(self) -> ProviderConcurrency:
        if self.concurrency is None:
            self.concurrency = ProviderConcurrency(self.config.runtime)
        return self.concurrency
