from __future__ import annotations

import json
from typing import Any


_REPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}


async def rephrase_tts_text(
    llm_provider: object,
    *,
    text: str,
    target_duration_ms: int,
    current_duration_ms: int,
    segment_id: str,
    protected_spans: list[dict[str, object]] | None = None,
    max_chars: int | None = None,
    required_compression_ratio: float | None = None,
) -> str:
    system_prompt = (
        "You shorten Vietnamese dubbing text for TTS timing. Return raw JSON only. "
        "Preserve the original meaning, technical terms, names, formulas, symbols, numbers, units, and domain terminology. "
        "Every protected span is binding: preserve its technical concept and use its spoken form; never use a forbidden rendering. "
        "Make the text shorter and easier to speak naturally in Vietnamese; remove redundancy, not information. "
        "Do not add new facts, commentary, greetings, markdown, or explanations. "
        "Return one concise text value that can be spoken inside the target duration. "
        "Never return an empty text value. "
        "If no safe shorter wording exists, return the original text unchanged. "
        "When max_chars is provided, the text must not exceed that hard character limit."
    )
    user_prompt = json.dumps(
        {
            "segment_id": segment_id,
            "target_duration_ms": target_duration_ms,
            "current_duration_ms": current_duration_ms,
            "max_chars": max_chars,
            "required_compression_ratio": required_compression_ratio,
            "instruction": (
                "Return a shorter natural Vietnamese sentence or phrase in the text field. "
                "Keep the same language and preserve all essential technical meaning. "
                "The text field must be non-empty. "
                + (f"Hard limit: at most {max_chars} characters. " if max_chars is not None else "")
            ),
            "text": text,
            "protected_spans": protected_spans or [],
        },
        ensure_ascii=False,
    )
    if hasattr(llm_provider, "complete_structured_json"):
        result = await llm_provider.complete_structured_json(
            system_prompt,
            user_prompt,
            _REPHRASE_SCHEMA,
            response_name="tts_rephrase_output",
            response_description="Shortened Vietnamese text for TTS timing.",
        )
    elif hasattr(llm_provider, "complete_json"):
        result = await llm_provider.complete_json(system_prompt, user_prompt, schema=_REPHRASE_SCHEMA)
    else:
        raise AttributeError("LLM provider does not support structured JSON completion")
    rephrased = str(result.get("text", "")).strip()
    if not rephrased:
        raise ValueError(f"{segment_id}: tts_rephrase_empty")
    return rephrased
