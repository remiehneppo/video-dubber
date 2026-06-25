from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from dubber.core.enums import StageName
from dubber.core.io import read_json, write_json_atomic
from dubber.core.models import utc_now_iso


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    version: int
    path: str
    sha256: str
    created_by_stage: StageName
    schema_version: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "sha256": self.sha256,
            "created_by_stage": self.created_by_stage.value,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ArtifactEntry:
        return cls(
            name=str(data["name"]),
            version=int(data["version"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            created_by_stage=StageName(str(data["created_by_stage"])),
            schema_version=str(data["schema_version"]),
            created_at=str(data["created_at"]),
        )


@dataclass
class ArtifactManifest:
    job_id: str
    manifest_path: Path
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    schema_version: str = "1.0"

    @classmethod
    def create(cls, job_id: str, manifest_path: Path) -> ArtifactManifest:
        return cls(job_id=job_id, manifest_path=manifest_path)

    @classmethod
    def load(cls, manifest_path: Path) -> ArtifactManifest:
        data = read_json(manifest_path)
        return cls(
            job_id=str(data["job_id"]),
            manifest_path=manifest_path,
            artifacts=[ArtifactEntry.from_dict(item) for item in data.get("artifacts", [])],
            schema_version=str(data.get("schema_version", "1.0")),
        )

    def record_artifact(
        self,
        *,
        name: str,
        version: int,
        path: Path,
        created_by_stage: StageName,
        schema_version: str,
    ) -> ArtifactEntry:
        relative_path = path.resolve().relative_to(self.manifest_path.parent.resolve()).as_posix()
        entry = ArtifactEntry(
            name=name,
            version=version,
            path=relative_path,
            sha256=sha256_file(path),
            created_by_stage=created_by_stage,
            schema_version=schema_version,
            created_at=utc_now_iso(),
        )
        self.artifacts = [
            existing
            for existing in self.artifacts
            if not (existing.name == name and existing.version == version)
        ]
        self.artifacts.append(entry)
        return entry

    def get(self, name: str, version: int) -> ArtifactEntry | None:
        for artifact in self.artifacts:
            if artifact.name == name and artifact.version == version:
                return artifact
        return None

    def validate_artifact(self, name: str, version: int) -> bool:
        artifact = self.get(name, version)
        if artifact is None:
            return False
        path = self.manifest_path.parent / artifact.path
        return path.exists() and sha256_file(path) == artifact.sha256

    def save(self) -> None:
        write_json_atomic(
            self.manifest_path,
            {
                "schema_version": self.schema_version,
                "job_id": self.job_id,
                "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            },
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

