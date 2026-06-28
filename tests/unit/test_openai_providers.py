from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from dubber.providers.asr_openai_compatible import OpenAICompatibleASRProvider
from dubber.providers.llm_openai_compatible import OpenAICompatibleLLMProvider
from dubber.providers.retry import ProviderRequestError
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


def test_openai_compatible_asr_requests_verbose_word_timestamps_and_disables_vad_filter(tmp_path: Path) -> None:
    audio = tmp_path / "seg.wav"
    audio.write_bytes(b"RIFFfake")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("latin-1")
        return httpx.Response(
            200,
            json={
                "text": "hello",
                "language": "en",
                "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            },
        )

    provider = OpenAICompatibleASRProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="whisper-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timestamp_mode="prefer_word",
        vad_filter=False,
    )

    result = asyncio.run(provider.transcribe(audio, language="en"))

    body = seen["body"]
    assert 'name="response_format"' in body
    assert "verbose_json" in body
    assert 'name="timestamp_granularities[]"' in body
    assert "word" in body
    assert 'name="vad_filter"' in body
    assert "false" in body
    assert result.raw["words"][0]["word"] == "hello"


def test_openai_compatible_asr_retries_transient_server_errors(tmp_path: Path) -> None:
    audio = tmp_path / "seg.wav"
    audio.write_bytes(b"RIFFfake")
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 5:
            return httpx.Response(500, json={"error": "temporary"})
        return httpx.Response(200, json={"text": "hello", "confidence": 0.87, "language": "en"})

    provider = OpenAICompatibleASRProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="whisper-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay_sec=0,
    )

    result = asyncio.run(provider.transcribe(audio, language="en"))

    assert attempts == 5
    assert result.text == "hello"


def test_openai_compatible_asr_reports_error_after_five_failed_attempts(tmp_path: Path) -> None:
    audio = tmp_path / "seg.wav"
    audio.write_bytes(b"RIFFfake")
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "still unavailable"})

    provider = OpenAICompatibleASRProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="whisper-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay_sec=0,
    )

    try:
        asyncio.run(provider.transcribe(audio, language="en"))
    except ProviderRequestError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ProviderRequestError")

    assert attempts == 5
    assert "asr request failed after 5 attempts" in message


def test_openai_compatible_asr_creates_fresh_client_for_separate_event_loops(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "seg.wav"
    audio.write_bytes(b"RIFFfake")

    class FakeAsyncClient:
        instances = 0
        closes = 0

        def __init__(self, timeout=None) -> None:
            type(self).instances += 1
            self.timeout = timeout
            self.closed = False

        async def post(self, url, headers=None, data=None, files=None):
            assert not self.closed
            return httpx.Response(
                200,
                json={"text": "hello", "confidence": 0.87, "language": "en"},
                request=httpx.Request("POST", url),
            )

        async def aclose(self) -> None:
            self.closed = True
            type(self).closes += 1

    monkeypatch.setattr("dubber.providers.asr_openai_compatible.httpx.AsyncClient", FakeAsyncClient)

    provider = OpenAICompatibleASRProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="whisper-1",
    )

    first = asyncio.run(provider.transcribe(audio, language="en"))
    second = asyncio.run(provider.transcribe(audio, language="en"))

    assert first.text == "hello"
    assert second.text == "hello"
    assert FakeAsyncClient.instances == 2
    assert FakeAsyncClient.closes == 2


def test_openai_compatible_llm_posts_chat_request_and_parses_json() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        payload = json.loads(request.content.decode("utf-8"))
        seen["payload"] = payload
        seen["response_format"] = payload["response_format"]
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
    assert seen["payload"]["temperature"] == 0  # type: ignore[index]
    system_message = seen["payload"]["messages"][0]["content"]  # type: ignore[index]
    assert "Return exactly one valid JSON object and nothing else" in system_message
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["name"] == "structured_output"
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert result == {"terms": [{"original": "eigenvector"}]}


def test_openai_compatible_llm_parses_streamed_sse_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            content=(
                b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"{\\"ok\\":"}}]}\n\n'
                b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" true}"}}]}\n\n'
                b'data: [DONE]\n\n'
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert result == {"ok": True}


def test_openai_compatible_llm_parses_streamed_sse_fenced_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"index":0,"delta":{"content":"```json\\n{\\"ok\\": true}\\n```"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n'
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert result == {"ok": True}


def test_openai_compatible_llm_parses_sse_body_without_stream_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"index":0,"delta":{"content":"```json\\n{\\"ok\\": true}\\n```"},"finish_reason":null}]}\n\n'
                b'data: [DONE]\n\n'
            ),
            headers={"content-type": "text/plain"},
            request=request,
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert result == {"ok": True}


def test_openai_compatible_llm_parses_message_parsed_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "parsed": {"ok": True}}}]},
            request=request,
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert result == {"ok": True}


def test_openai_compatible_llm_parses_content_parts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"type": "text", "text": "{\"ok\":"}, {"type": "text", "text": " true}"}]}}]},
            request=request,
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert result == {"ok": True}


def test_openai_compatible_llm_reports_empty_content_with_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
            request=request,
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))
    except ValueError as exc:
        assert "LLM response was not parseable JSON" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_openai_compatible_llm_retries_transient_server_errors() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 5:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="gpt-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay_sec=0,
    )

    result = asyncio.run(provider.complete_json("system", "user", schema={"type": "object"}))

    assert attempts == 5
    assert result == {"ok": True}


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


def test_openai_compatible_tts_retries_transient_server_errors(tmp_path: Path) -> None:
    output = tmp_path / "tts.wav"
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 5:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, content=b"WAVDATA", headers={"content-type": "audio/wav"})

    provider = OpenAICompatibleTTSProvider(
        base_url="https://api.example.test/v1",
        api_key="secret",
        model="tts-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay_sec=0,
    )

    result = asyncio.run(provider.synthesize("xin chao", voice="nova", output_path=output))

    assert attempts == 5
    assert output.read_bytes() == b"WAVDATA"
    assert result.audio_path == output
