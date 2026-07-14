from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    input_dir: Path
    audio_dir: Path
    raw_dir: Path
    artifacts_dir: Path
    tts_dir: Path
    output_dir: Path
    logs_dir: Path
    job_state_file: Path
    manifest_file: Path

    @classmethod
    def create(cls, workspace_dir: Path, job_id: str, *, create_dirs: bool = True) -> WorkspacePaths:
        root = workspace_dir / job_id
        paths = cls(
            root=root,
            input_dir=root / "input",
            audio_dir=root / "audio",
            raw_dir=root / "raw",
            artifacts_dir=root / "artifacts",
            tts_dir=root / "tts",
            output_dir=root / "output",
            logs_dir=root / "logs",
            job_state_file=root / "job_state.json",
            manifest_file=root / "manifest.json",
        )
        if not create_dirs:
            return paths
        for directory in (
            paths.input_dir,
            paths.audio_dir,
            paths.raw_dir,
            paths.artifacts_dir,
            paths.tts_dir,
            paths.output_dir,
            paths.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths

    def artifact_path(self, name: str) -> Path:
        return self.resolve_relative(Path("artifacts") / name)

    def resolve_relative(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative path: {relative_path}")
        resolved = (self.root / relative).resolve()
        root = self.root.resolve()
        if root not in (resolved, *resolved.parents):
            raise ValueError(f"Unsafe relative path: {relative_path}")
        return resolved

    def to_relative(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.root.resolve()
        if root not in (resolved, *resolved.parents):
            raise ValueError(f"Path is outside workspace: {path}")
        return resolved.relative_to(root).as_posix()

