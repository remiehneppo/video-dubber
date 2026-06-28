from __future__ import annotations

from dubber.tts.duration_planner import plan_segment_duration


def test_duration_planner_slows_down_when_tts_is_shorter() -> None:
    plan = plan_segment_duration("seg_000001", orig_duration_ms=1000, tts_duration_ms=800)

    assert plan.action == "time_stretch"
    assert plan.stretch_ratio == 0.8
    assert plan.overflow_ms == 0
    assert plan.warnings == ["tts_duration_stretched_to_original"]


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
    assert plan.warnings == ["tts_duration_overflow"]
