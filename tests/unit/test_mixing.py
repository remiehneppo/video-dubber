from __future__ import annotations

import pytest

from dubber.mixing.ducking import build_commentary_filter, db_to_linear


def test_db_to_linear_converts_negative_db_to_gain() -> None:
    assert db_to_linear(-20) == pytest.approx(0.1, rel=0.01)
    assert db_to_linear(-6) == pytest.approx(0.501, rel=0.01)


def test_build_commentary_filter_ducks_original_and_normalizes_loudness() -> None:
    filter_graph = build_commentary_filter(original_ducking_db=-22, tts_boost_db=8.0, final_loudness_normalization=True)

    assert "[0:a]volume=0.079433" not in filter_graph
    assert "volume=2.511886" in filter_graph
    assert "asplit=2[tts][sidechain]" in filter_graph
    assert "sidechaincompress=threshold=0.079433" in filter_graph
    assert "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0" in filter_graph
    assert "loudnorm=I=-16" in filter_graph
    assert filter_graph.endswith("[out]")


def test_build_commentary_filter_can_skip_loudness_normalization() -> None:
    filter_graph = build_commentary_filter(original_ducking_db=-22, tts_boost_db=8.0, final_loudness_normalization=False)

    assert "loudnorm" not in filter_graph
    assert filter_graph.endswith("[out]")
