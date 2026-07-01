from __future__ import annotations

from pathlib import Path
import wave

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
        request_timeout_sec: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.client = client
        self.request_timeout_sec = request_timeout_sec
        self.max_attempts = max_attempts
        self.retry_delay_sec = retry_delay_sec

    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        client = self.client or httpx.AsyncClient(timeout=self.request_timeout_sec)
        created_client = self.client is None

        async def request() -> httpx.Response:
            return await client.post(
                f"{self.base_url}/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": text, "voice": voice},
            )

        try:
            response = await request_with_retries(
                request,
                provider="tts",
                max_attempts=self.max_attempts,
                retry_delay_sec=self.retry_delay_sec,
            )
        finally:
            if created_client:
                await client.aclose()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("audio/"):
            raise ValueError(f"TTS response has invalid audio content type: {content_type or 'missing'}")
        if not response.content:
            raise ValueError("TTS response body is empty")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path = output_path.with_name(f".{output_path.name}.provider-attempt")
        attempt_path.write_bytes(response.content)
        try:
            with wave.open(str(attempt_path), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getframerate() <= 0 or wav.getnframes() <= 0:
                    raise ValueError("TTS WAV has invalid PCM format or duration")
                duration_ms = int(wav.getnframes() * 1000 / wav.getframerate())
        except (wave.Error, EOFError, ValueError) as exc:
            attempt_path.unlink(missing_ok=True)
            raise ValueError("TTS response is not a decodable WAV") from exc
        attempt_path.replace(output_path)
        return TTSResult(
            audio_path=output_path,
            duration_ms=duration_ms,
            provider_metadata={
                "content_type": content_type,
                "response_bytes": len(response.content),
                "duration_ms": duration_ms,
            },
        )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
