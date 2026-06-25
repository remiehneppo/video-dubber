from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ASRResult:
    text: str
    confidence: float | None
    language: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class TTSResult:
    audio_path: Path
    duration_ms: int | None
    provider_metadata: dict[str, Any]
