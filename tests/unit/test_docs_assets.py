from __future__ import annotations

from pathlib import Path


def test_readme_documents_core_commands_and_mock_run() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "dubber run" in readme
    assert "--provider-mode mock" in readme
    assert "dubber resume" in readme
    assert "dubber web" in readme


def test_env_example_lists_provider_secrets_without_real_values() -> None:
    env = Path(".env.example").read_text(encoding="utf-8")

    for key in ["ASR_BASE_URL", "ASR_API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "TTS_BASE_URL", "TTS_API_KEY"]:
        assert key in env
    assert "sk-" not in env


def test_runbook_documents_common_recovery_paths() -> None:
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")

    assert "Glossary review" in runbook
    assert "Crash during TTS" in runbook
    assert "Corrupt artifact" in runbook
    assert "rerun-segment" in runbook
