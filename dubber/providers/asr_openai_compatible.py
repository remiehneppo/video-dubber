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
        timestamp_mode: str = "prefer_word",
        vad_filter: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = client
        self.request_timeout_sec = request_timeout_sec
        self.max_attempts = max_attempts
        self.retry_delay_sec = retry_delay_sec
        self.timestamp_mode = timestamp_mode
        self.vad_filter = vad_filter

    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        client = self.client or httpx.AsyncClient(timeout=self.request_timeout_sec)
        created_client = self.client is None

        async def request() -> httpx.Response:
            with audio_path.open("rb") as audio_file:
                return await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=self._request_data(language),
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                )

        try:
            response = await request_with_retries(
                request,
                provider="asr",
                max_attempts=self.max_attempts,
                retry_delay_sec=self.retry_delay_sec,
            )
        finally:
            if created_client:
                await client.aclose()
        payload = response.json()
        return ASRResult(
            text=str(payload.get("text", "")),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            language=str(payload["language"]) if payload.get("language") is not None else language,
            raw=payload,
        )

    def _request_data(self, language: str) -> dict[str, str]:
        data = {
            "model": self.model,
            "language": language,
            "response_format": "verbose_json",
            "vad_filter": "true" if self.vad_filter else "false",
        }
        if self.timestamp_mode in {"prefer_word", "word"}:
            data["timestamp_granularities[]"] = "word"
        return data

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
