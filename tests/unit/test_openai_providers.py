from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from dubber.providers.asr_openai_compatible import OpenAICompatibleASRProvider
from dubber.providers.llm_openai_compatible import OpenAICompatibleLLMProvider
from dubber.providers.tts_openai_compatible import OpenAICompatibleTTSProvider


def test_openai_compatible_asr_posts_audio_and_normalizes_result(tmp_path: Path) -> None:
    audio = tmp_path / "seg.wav"
    audio.write_bytes(b"RIFFfake")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["content_type"] = request.headers.get("content-type")
        return httpx.Response(200, json={"text": "hello", "confidence": 0.87, "language": "en"})

    provider = OpenAICompatibleASRProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="whisper-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.transcribe(audio, language="en"))

    assert seen["url"] == "https://api.example.test/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer secret"
    assert str(seen["content_type"]).startswith("multipart/form-data")
    assert result.text == "hello"
    assert result.confidence == 0.87
    assert result.language == "en"


def test_openai_compatible_llm_posts_chat_request_and_parses_json() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        payload = json.loads(request.content.decode("utf-8"))
        seen["payload"] = payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"terms": [{"original": "eigenvector"}]}'}}
                ]
            },
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["payload"]["model"] == "gpt-test"  # type: ignore[index]
    assert result == {"terms": [{"original": "eigenvector"}]}


def test_openai_compatible_tts_writes_audio_file(tmp_path: Path) -> None:
    output = tmp_path / "tts.wav"
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=b"WAVDATA", headers={"content-type": "audio/wav"})

    provider = OpenAICompatibleTTSProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="tts-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.synthesize("xin chao", voice="nova", output_path=output))

    assert seen["url"] == "https://api.example.test/v1/audio/speech"
    assert seen["payload"] == {"model": "tts-1", "input": "xin chao", "voice": "nova"}
    assert output.read_bytes() == b"WAVDATA"
    assert result.audio_path == output
