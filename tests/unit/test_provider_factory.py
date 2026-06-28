from __future__ import annotations

from dubber.core.models import ASRServiceConfig, DubberConfig, LLMServiceConfig, RuntimeConfig, TTSServiceConfig
from dubber.providers.asr_openai_compatible import OpenAICompatibleASRProvider
from dubber.providers.factory import ProviderConfigError, build_provider_bundle, validate_provider_config
from dubber.providers.llm_openai_compatible import OpenAICompatibleLLMProvider
from dubber.providers.tts_openai_compatible import OpenAICompatibleTTSProvider


def test_build_provider_bundle_creates_openai_compatible_adapters() -> None:
    config = DubberConfig(
        asr_service=ASRServiceConfig(base_url="https://asr.example/v1", api_key="asr", model="whisper"),
        llm_service=LLMServiceConfig(base_url="https://llm.example/v1", api_key="llm", model="gpt"),
        tts_service=TTSServiceConfig(base_url="https://tts.example/v1", api_key="tts", model="tts", voice="nova"),
    )

    bundle = build_provider_bundle(config)

    assert isinstance(bundle.asr, OpenAICompatibleASRProvider)
    assert isinstance(bundle.llm, OpenAICompatibleLLMProvider)
    assert isinstance(bundle.tts, OpenAICompatibleTTSProvider)
    assert bundle.tts.voice == "nova"


def test_validate_provider_config_reports_missing_secrets() -> None:
    config = DubberConfig(
        asr_service=ASRServiceConfig(base_url="", api_key=""),
        llm_service=LLMServiceConfig(base_url="https://llm.example/v1", api_key=""),
        tts_service=TTSServiceConfig(base_url="https://tts.example/v1", api_key="tts"),
    )

    try:
        validate_provider_config(config)
    except ProviderConfigError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ProviderConfigError")

    assert "asr_service.base_url" in message
    assert "asr_service.api_key" in message
    assert "llm_service.api_key" in message


def test_build_provider_bundle_applies_runtime_retry_and_timeout() -> None:
    config = DubberConfig(
        runtime=RuntimeConfig(
            retry_max_attempts=7,
            retry_backoff_sec=3,
            request_timeout_sec=600,
        ),
        asr_service=ASRServiceConfig(base_url="https://asr.example/v1", api_key="asr", model="whisper"),
        llm_service=LLMServiceConfig(base_url="https://llm.example/v1", api_key="llm", model="gpt"),
        tts_service=TTSServiceConfig(base_url="https://tts.example/v1", api_key="tts", model="tts", voice="nova"),
    )

    bundle = build_provider_bundle(config)

    for provider in (bundle.asr, bundle.llm, bundle.tts):
        assert provider.max_attempts == 7
        assert provider.retry_delay_sec == 3
        assert provider.request_timeout_sec == 600
