from __future__ import annotations

from pathlib import Path

import httpx

from dubber.providers.base import ASRResult


class OpenAICompatibleASRProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.AsyncClient(timeout=120)

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        with audio_path.open("rb") as audio_file:
            response = await self.client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"model": self.model, "language": language},
                files={"file": (audio_path.name, audio_file, "audio/wav")},
            )
        response.raise_for_status()
        payload = response.json()
        return ASRResult(
            text=str(payload.get("text", "")),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            language=str(payload["language"]) if payload.get("language") is not None else language,
            raw=payload,
        )

    async def close(self) -> None:
        await self.client.aclose()
