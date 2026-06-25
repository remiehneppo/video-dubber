from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dubber.core.models import DubberConfig
from dubber.providers.asr_openai_compatible import OpenAICompatibleASRProvider
from dubber.providers.llm_openai_compatible import OpenAICompatibleLLMProvider
from dubber.providers.tts_openai_compatible import OpenAICompatibleTTSProvider


class ProviderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderBundle:
    asr: Any
    llm: Any
    tts: Any


def validate_provider_config(config: DubberConfig) -> None:
    missing: list[str] = []
    for prefix, service in (
        ("asr_service", config.asr_service),
        ("llm_service", config.llm_service),
        ("tts_service", config.tts_service),
    ):
        if service.provider != "openai_compatible":
            raise ProviderConfigError(f"Unsupported {prefix}.provider: {service.provider}")
        if _is_missing(service.base_url):
            missing.append(f"{prefix}.base_url")
        if _is_missing(service.api_key):
            missing.append(f"{prefix}.api_key")
    if missing:
        raise ProviderConfigError("provider config invalid; missing " + ", ".join(missing))


def build_provider_bundle(config: DubberConfig) -> ProviderBundle:
    validate_provider_config(config)
    return ProviderBundle(
        asr=OpenAICompatibleASRProvider(
            base_url=config.asr_service.base_url,
            api_key=config.asr_service.api_key,
            model=config.asr_service.model,
        ),
        llm=OpenAICompatibleLLMProvider(
            base_url=config.llm_service.base_url,
            api_key=config.llm_service.api_key,
            model=config.llm_service.model,
        ),
        tts=OpenAICompatibleTTSProvider(
            base_url=config.tts_service.base_url,
            api_key=config.tts_service.api_key,
            model=config.tts_service.model,
        ),
    )


def _is_missing(value: str) -> bool:
    stripped = value.strip()
    return not stripped or (stripped.startswith("${") and stripped.endswith("}"))
