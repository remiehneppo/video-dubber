from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from dubber.providers.retry import request_with_retries

logger = logging.getLogger(__name__)


class LLMStructuredOutputError(ValueError):
    def __init__(self, message: str, *, content: str, body: str) -> None:
        super().__init__(message)
        self.content = content
        self.body = body


class OpenAICompatibleLLMProvider:
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
        self.client = client
        self.request_timeout_sec = request_timeout_sec
        self.max_attempts = max_attempts
        self.retry_delay_sec = retry_delay_sec

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return await self.complete_structured_json(
            system_prompt,
            user_prompt,
            schema,
            response_name="structured_output",
            response_description="Structured JSON response",
        )

    async def complete_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        *,
        response_name: str,
        response_description: str,
    ) -> dict[str, Any]:
        client = self.client or httpx.AsyncClient(timeout=self.request_timeout_sec)
        created_client = self.client is None
        structured_system_prompt = self._structured_system_prompt(
            system_prompt,
            schema=schema,
            response_name=response_name,
        )

        async def request() -> httpx.Response:
            return await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": structured_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_name,
                            "description": response_description,
                            "schema": schema,
                            "strict": True,
                        },
                    },
                },
            )

        try:
            response = await request_with_retries(
                request,
                provider="llm",
                max_attempts=self.max_attempts,
                retry_delay_sec=self.retry_delay_sec,
            )
        finally:
            if created_client:
                await client.aclose()
        content = self._extract_content(response)
        try:
            parsed = self._loads_json(content)
        except (json.JSONDecodeError, ValueError) as exc:
            preview = self._preview(content)
            body_preview = self._preview(response.text)
            logger.error("llm response was not parseable JSON content_preview=%r body_preview=%r", preview, body_preview)
            raise LLMStructuredOutputError(
                f"LLM response was not parseable JSON: {preview}",
                content=content,
                body=response.text,
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object")
        return parsed

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def _extract_content(self, response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type and not response.text.lstrip().startswith("data:"):
            payload = response.json()
            choice = payload["choices"][0]
            message = choice.get("message", {})
            if isinstance(message, dict):
                parsed = message.get("parsed")
                if parsed is not None:
                    return json.dumps(parsed, ensure_ascii=False)
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(self._content_part_text(part) for part in content)
            return ""

        chunks: list[str] = []
        for event in response.text.split("\n\n"):
            event = event.strip()
            if not event:
                continue
            data_lines = [line.removeprefix("data:").strip() for line in event.splitlines() if line.startswith("data:")]
            for data in data_lines:
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                choices = payload.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if isinstance(delta, dict) and delta.get("content"):
                    chunks.append(str(delta["content"]))
                message = choices[0].get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    chunks.append(str(message["content"]))
        content = "".join(chunks).strip()
        if not content:
            logger.error("llm stream response did not include completion content body_preview=%r", self._preview(response.text))
            raise ValueError("LLM stream response did not include completion content")
        return content

    def _loads_json(self, content: str) -> Any:
        content = content.strip()
        if not content:
            raise ValueError("LLM response content is empty")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            fenced = self._extract_fenced_json(content)
            if fenced is not None:
                return json.loads(fenced)
            embedded = self._extract_embedded_json(content)
            if embedded is not None:
                return embedded
            raise

    def _extract_fenced_json(self, content: str) -> str | None:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                body = "\n".join(lines[1:-1]).strip()
                if body.startswith("json"):
                    body = body[4:].lstrip()
                return body or None
        return None

    def _extract_embedded_json(self, content: str) -> Any | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(content):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            return parsed
        return None

    def _content_part_text(self, part: object) -> str:
        if isinstance(part, str):
            return part
        if not isinstance(part, dict):
            return ""
        text = part.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, dict) and isinstance(text.get("value"), str):
            return str(text["value"])
        return ""

    def _preview(self, value: str, limit: int = 500) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    def _structured_system_prompt(self, system_prompt: str, *, schema: dict[str, Any], response_name: str) -> str:
        schema_text = json.dumps(schema, ensure_ascii=False)
        return (
            f"{system_prompt}\n\n"
            "Structured output contract:\n"
            f"- Response name: {response_name}\n"
            "- Return exactly one valid JSON object and nothing else.\n"
            "- Do not include markdown fences, prose, explanations, headings, or bullet lists.\n"
            "- The first non-whitespace character must be { and the last non-whitespace character must be }.\n"
            "- The JSON object must conform to this JSON Schema:\n"
            f"{schema_text}"
        )
