from __future__ import annotations

from pathlib import Path

import pytest

from dubber.core.config import load_config
from dubber.core.models import ASRChunkingConfig


def test_asr_chunking_is_enabled_by_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")

    config = load_config(config_file)

    assert ASRChunkingConfig().enabled is True
    assert config.asr_chunking.enabled is True


def test_load_config_allows_explicit_asr_chunking_legacy_disable(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("asr_chunking:\n  enabled: false\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.asr_chunking.enabled is False


def test_load_config_applies_asr_chunking_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "asr_chunking:",
                "  enabled: true",
                "  max_chunk_duration_ms: 45000",
                "  initial_silence_ms: 4000",
                "  min_silence_ms: 1000",
                "  silence_step_ms: 250",
                "  trailing_silence_cap_ms: 3000",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.asr_chunking.enabled is True
    assert config.asr_chunking.max_chunk_duration_ms == 45_000
    assert config.asr_chunking.initial_silence_ms == 4_000
    assert config.asr_chunking.min_silence_ms == 1_000
    assert config.asr_chunking.silence_step_ms == 250
    assert config.asr_chunking.trailing_silence_cap_ms == 3_000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_chunk_duration_ms", 0),
        ("initial_silence_ms", -1),
        ("min_silence_ms", 0),
        ("silence_step_ms", True),
        ("trailing_silence_cap_ms", 0),
    ],
)
def test_load_config_rejects_non_positive_asr_chunking_values(
    tmp_path: Path,
    name: str,
    value: object,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"asr_chunking:\n  {name}: {str(value).lower()}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"asr_chunking\.{name} must be an integer >= 1"):
        load_config(config_file)


def test_load_config_rejects_asr_chunking_initial_below_minimum(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "asr_chunking:",
                "  initial_silence_ms: 500",
                "  min_silence_ms: 1000",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="asr_chunking.initial_silence_ms must be >= asr_chunking.min_silence_ms",
    ):
        load_config(config_file)
