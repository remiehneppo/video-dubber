from __future__ import annotations

from dubber.tts.duration_planner import plan_segment_duration


def test_duration_planner_slows_down_when_tts_is_shorter() -> None:
    plan = plan_segment_duration("seg_000001", orig_duration_ms=1000, tts_duration_ms=800)

    assert plan.action == "pad_silence"
    assert plan.stretch_ratio == 1.0
    assert plan.overflow_ms == 0
    assert plan.warnings == ["tts_duration_shorter_than_source"]


def test_duration_planner_pads_when_tts_matches_original_duration() -> None:
    plan = plan_segment_duration("seg_000001", orig_duration_ms=1000, tts_duration_ms=1000)

    assert plan.action == "pad_silence"
    assert plan.stretch_ratio == 1.0


def test_duration_planner_stretches_slightly_long_audio() -> None:
    plan = plan_segment_duration("seg_000001", orig_duration_ms=1000, tts_duration_ms=1100)

    assert plan.action == "time_stretch"
    assert plan.stretch_ratio == 1.1
    assert plan.warnings == []


def test_duration_planner_warns_near_hard_limit() -> None:
    plan = plan_segment_duration("seg_000001", orig_duration_ms=1000, tts_duration_ms=1250)

    assert plan.action == "time_stretch"
    assert plan.stretch_ratio == 1.25
    assert plan.warnings == ["tts_duration_near_hard_limit"]


def test_duration_planner_requests_rephrase_above_hard_limit() -> None:
    plan = plan_segment_duration("seg_000001", orig_duration_ms=1000, tts_duration_ms=1400)

    assert plan.action == "rephrase"
    assert plan.stretch_ratio == 1.0
    assert plan.warnings == ["tts_duration_exceeds_hard_limit"]


def test_duration_planner_allows_controlled_overflow_after_rephrase_failure() -> None:
    plan = plan_segment_duration(
        "seg_000001",
        orig_duration_ms=1000,
        tts_duration_ms=1500,
        rephrase_already_attempted=True,
        max_overflow_ms=800,
    )

    assert plan.action == "overflow"
    assert plan.overflow_ms == 500
    assert plan.warnings == ["tts_duration_overflow_into_source_silence"]


def test_duration_planner_allows_overflow_after_rephrase_when_gap_before_next_segment_allows_it() -> None:
    plan = plan_segment_duration(
        "seg_000001",
        orig_duration_ms=1000,
        tts_duration_ms=1800,
        rephrase_already_attempted=True,
        max_overflow_ms=900,
    )

    assert plan.action == "overflow"
    assert plan.overflow_ms == 800
    assert plan.warnings == ["tts_duration_overflow_into_source_silence"]


def test_duration_planner_prefers_overflow_into_available_source_silence_over_speedup() -> None:
    plan = plan_segment_duration(
        "seg_000001",
        orig_duration_ms=1000,
        tts_duration_ms=1150,
        max_overflow_ms=200,
    )

    assert plan.action == "overflow"
    assert plan.stretch_ratio == 1.0
    assert plan.overflow_ms == 150
    assert plan.warnings == ["tts_duration_overflow_into_source_silence"]



def test_duration_planner_combines_source_duration_and_trailing_silence_before_speedup() -> None:
    plan = plan_segment_duration(
        "seg_000010",
        orig_duration_ms=3220,
        tts_duration_ms=7420,
        speedup_hard_limit=1.3,
        max_overflow_ms=3840,
    )

    assert plan.action == "time_stretch_overflow"
    assert plan.stretch_ratio == 1.051
    assert plan.overflow_ms == 3840
    assert plan.warnings == ["tts_duration_uses_source_silence_and_time_stretch"]
