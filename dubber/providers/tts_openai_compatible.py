from __future__ import annotations

from pathlib import Path

import httpx

from dubber.providers.base import TTSResult
from dubber.providers.retry import request_with_retries


class OpenAICompatibleTTSProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        voice: str = "default",
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 5,
        retry_delay_sec: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.client = client or httpx.AsyncClient(timeout=120)
        self.max_attempts = max_attempts
        self.retry_delay_sec = retry_delay_sec

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        response = await request_with_retries(
            lambda: self.client.post(
                f"{self.base_url}/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": text, "voice": voice},
            ),
            provider="tts",
            max_attempts=self.max_attempts,
            retry_delay_sec=self.retry_delay_sec,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return TTSResult(audio_path=output_path, duration_ms=None, provider_metadata={"content_type": response.headers.get("content-type")})

    async def close(self) -> None:
        await self.client.aclose()
