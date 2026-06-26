from __future__ import annotations

from pathlib import Path

import httpx

from dubber.providers.base import ASRResult
from dubber.providers.retry import request_with_retries


class OpenAICompatibleASRProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 5,
        retry_delay_sec: float = 0.5,
        request_timeout_sec: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.AsyncClient(timeout=request_timeout_sec)
        self.max_attempts = max_attempts
        self.retry_delay_sec = retry_delay_sec

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        async def request() -> httpx.Response:
            with audio_path.open("rb") as audio_file:
                return await self.client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data={"model": self.model, "language": language},
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                )

        response = await request_with_retries(
            request,
            provider="asr",
            max_attempts=self.max_attempts,
            retry_delay_sec=self.retry_delay_sec,
        )
        payload = response.json()
        return ASRResult(
            text=str(payload.get("text", "")),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            language=str(payload["language"]) if payload.get("language") is not None else language,
            raw=payload,
        )

    async def close(self) -> None:
        await self.client.aclose()
