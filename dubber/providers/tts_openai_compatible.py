from __future__ import annotations

from pathlib import Path

import httpx

from dubber.providers.base import TTSResult


class OpenAICompatibleTTSProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.AsyncClient(timeout=120)

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        response = await self.client.post(
            f"{self.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text, "voice": voice},
        )
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return TTSResult(audio_path=output_path, duration_ms=None, provider_metadata={"content_type": response.headers.get("content-type")})

    async def close(self) -> None:
        await self.client.aclose()
