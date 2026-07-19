from __future__ import annotations

from pathlib import Path


def test_readme_documents_core_commands_and_mock_run() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "dubber run" in readme
    assert "--provider-mode mock" in readme
    assert "dubber resume" in readme
    assert "dubber web" in readme
    assert "Domain And Domain Profile Files" in readme
    assert "dubber/domain/<profile_id>.yaml" in readme


def test_env_example_lists_provider_secrets_without_real_values() -> None:
    env = Path(".env.example").read_text(encoding="utf-8")

    for key in ["ASR_BASE_URL", "ASR_API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "TTS_BASE_URL", "TTS_API_KEY"]:
        assert key in env
    assert "sk-" not in env


def test_runbook_documents_common_recovery_paths() -> None:
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")

    assert "Glossary review" in runbook
    assert "Domain profile setup" in runbook
    assert "Crash during TTS" in runbook
    assert "Corrupt artifact" in runbook
    assert "rerun-segment" in runbook


def test_config_example_documents_active_vad_asr_knobs() -> None:
    example = Path("config.example.yaml").read_text(encoding="utf-8")

    vad_block = example.split("translation:", 1)[0].split("vad:", 1)[1]

    assert "mode-specific" in example
    assert "không tác động đồng thời" in example
    assert "min_duration_ms" not in vad_block
    assert "max_vad_chunk_ms" not in vad_block
    assert "vad_filter" in example
    assert "không tự đổi output của VAD stage" in example


def test_local_openai_profile_uses_current_vad_asr_defaults() -> None:
    profile = Path("configs/profiles/local-openai-compatible.yaml").read_text(encoding="utf-8")

    vad_block = profile.split("translation:", 1)[0].split("vad:", 1)[1]

    assert "min_duration_ms" not in vad_block
    assert "max_duration_ms" not in vad_block
    assert "max_vad_chunk_ms" not in vad_block
    assert "asr_chunking:" in profile
    assert "  enabled: true" in profile
    assert "  target_duration_ms: 8000" in profile
    assert "  min_duration_ms: 2500" in profile
    assert "  max_duration_ms: 12000" in profile
