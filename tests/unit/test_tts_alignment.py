from __future__ import annotations

from dubber.tts.aligner import build_atempo_chain


def test_build_atempo_chain_keeps_single_supported_ratio() -> None:
    assert build_atempo_chain(1.25) == "atempo=1.250000"


def test_build_atempo_chain_splits_large_speedup_into_supported_filters() -> None:
    assert build_atempo_chain(3.0) == "atempo=2.000000,atempo=1.500000"


def test_build_atempo_chain_splits_large_slowdown_into_supported_filters() -> None:
    assert build_atempo_chain(0.25) == "atempo=0.500000,atempo=0.500000"


def test_build_atempo_chain_rejects_non_positive_ratio() -> None:
    try:
        build_atempo_chain(0)
    except ValueError as exc:
        assert "ratio must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
