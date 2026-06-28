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
) -> str:
    system_prompt = (
        "Rewrite Vietnamese dubbing text so it can be spoken naturally inside the target duration. "
        "Preserve meaning, technical terms, names, numbers, and domain terminology. Return JSON only."
    )
    user_prompt = json.dumps(
        {
            "segment_id": segment_id,
            "target_duration_ms": target_duration_ms,
            "current_duration_ms": current_duration_ms,
            "instruction": "Return a shorter natural Vietnamese sentence or phrase in the text field.",
            "text": text,
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
