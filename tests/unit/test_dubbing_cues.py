from __future__ import annotations

from dubber.transcript.cues import build_dubbing_cues


def test_cue_planner_builds_stable_three_to_six_second_cues_at_natural_boundaries() -> None:
    words = []
    for index in range(50):
        start_ms = index * 500 + (800 if index >= 20 else 0)
        text = f"word{index}"
        if index in {9, 19, 31, 41, 49}:
            text += "."
        words.append({"text": text, "start_ms": start_ms, "end_ms": start_ms + 350})
    transcript = [{
        "segment_id": "seg_000001",
        "start_ms": 0,
        "end_ms": words[-1]["end_ms"],
        "duration_ms": words[-1]["end_ms"],
        "source_text": " ".join(word["text"] for word in words),
        "words": words,
    }]

    first = build_dubbing_cues(transcript, target_duration_ms=4000, min_duration_ms=1500, max_duration_ms=6000)
    second = build_dubbing_cues(transcript, target_duration_ms=4000, min_duration_ms=1500, max_duration_ms=6000)

    assert first == second
    assert 4 <= len(first) <= 7
    assert all(1500 <= cue["duration_ms"] <= 6000 for cue in first)
    assert all(cue["cue_id"].startswith("cue_") for cue in first)
    assert all(cue["parent_segment_ids"] == ["seg_000001"] for cue in first)
    assert any(cue["source_text"].endswith(".") for cue in first[:-1])


def test_cue_planner_rebalances_a_short_tail_without_exceeding_maximum() -> None:
    words = [
        {"text": f"word{index}", "start_ms": index * 500, "end_ms": index * 500 + 300}
        for index in range(13)
    ]
    cues = build_dubbing_cues([{
        "segment_id": "seg_1",
        "start_ms": 0,
        "end_ms": words[-1]["end_ms"],
        "duration_ms": words[-1]["end_ms"],
        "source_text": " ".join(word["text"] for word in words),
        "words": words,
    }])

    assert all(1500 <= cue["duration_ms"] <= 6000 for cue in cues)


def test_cue_planner_merges_short_middle_cue_when_safe_without_exceeding_maximum() -> None:
    words = [
        {"text": "a.", "start_ms": 0, "end_ms": 1500},
        {"text": "b", "start_ms": 2000, "end_ms": 2300},
        {"text": "c", "start_ms": 2400, "end_ms": 2700},
        {"text": "d", "start_ms": 2800, "end_ms": 3100},
        {"text": "e.", "start_ms": 3600, "end_ms": 5100},
        {"text": "f.", "start_ms": 5600, "end_ms": 7100},
    ]

    cues = build_dubbing_cues(
        [{
            "segment_id": "seg_middle_short",
            "start_ms": 0,
            "end_ms": words[-1]["end_ms"],
            "duration_ms": words[-1]["end_ms"],
            "source_text": " ".join(word["text"] for word in words),
            "words": words,
        }],
        target_duration_ms=2200,
        min_duration_ms=1200,
        max_duration_ms=3000,
    )

    assert all(1200 <= cue["duration_ms"] <= 3000 for cue in cues)
    assert not any(cue["source_text"] == "b" for cue in cues)
    assert any(" b" in str(cue["source_text"]) or str(cue["source_text"]).startswith("b ") for cue in cues)


def test_cue_planner_does_not_split_inside_calculus_formula_at_pause() -> None:
    words = [
        {"text": "the", "start_ms": 0, "end_ms": 260},
        {"text": "area", "start_ms": 400, "end_ms": 660},
        {"text": "is", "start_ms": 800, "end_ms": 1060},
        {"text": "2", "start_ms": 1200, "end_ms": 1460},
        {"text": "pi", "start_ms": 1600, "end_ms": 1860},
        {"text": "r", "start_ms": 2000, "end_ms": 2260},
        {"text": "dr", "start_ms": 3200, "end_ms": 3460},
        {"text": "so", "start_ms": 3600, "end_ms": 3860},
        {"text": "we", "start_ms": 4000, "end_ms": 4260},
        {"text": "integrate.", "start_ms": 4400, "end_ms": 4660},
        {"text": "Then", "start_ms": 5400, "end_ms": 5660},
        {"text": "continue", "start_ms": 5800, "end_ms": 6060},
        {"text": "with", "start_ms": 6200, "end_ms": 6460},
        {"text": "bounds.", "start_ms": 6600, "end_ms": 6860},
    ]

    cues = build_dubbing_cues(
        [{
            "segment_id": "seg_formula",
            "start_ms": 0,
            "end_ms": words[-1]["end_ms"],
            "duration_ms": words[-1]["end_ms"],
            "source_text": " ".join(word["text"] for word in words),
            "words": words,
        }],
        target_duration_ms=2400,
        min_duration_ms=1200,
        max_duration_ms=5000,
    )

    joined = " | ".join(str(cue["source_text"]) for cue in cues)
    assert "2 pi r | dr" not in joined
    assert any("2 pi r dr" in str(cue["source_text"]) for cue in cues)


def test_cue_planner_keeps_formula_intact_when_safe_boundary_exceeds_maximum() -> None:
    words = [
        {"text": "2", "start_ms": 0, "end_ms": 700},
        {"text": "pi", "start_ms": 800, "end_ms": 1500},
        {"text": "r", "start_ms": 1600, "end_ms": 2300},
        {"text": "dr", "start_ms": 2400, "end_ms": 3100},
        {"text": "therefore.", "start_ms": 3200, "end_ms": 3900},
        {"text": "continue.", "start_ms": 4300, "end_ms": 5000},
    ]

    cues = build_dubbing_cues(
        [{
            "segment_id": "seg_long_formula",
            "start_ms": 0,
            "end_ms": words[-1]["end_ms"],
            "duration_ms": words[-1]["end_ms"],
            "source_text": " ".join(word["text"] for word in words),
            "words": words,
        }],
        target_duration_ms=1800,
        min_duration_ms=1000,
        max_duration_ms=2200,
    )

    joined = " | ".join(str(cue["source_text"]) for cue in cues)
    assert "2 | pi" not in joined
    assert "pi | r" not in joined
    assert "r | dr" not in joined
    formula_cue = next(cue for cue in cues if "2 pi r dr" in str(cue["source_text"]))
    assert formula_cue["duration_ms"] > 2200
    assert "cue_duration_exceeds_max_for_safe_boundary" in formula_cue["risk_flags"]


def test_cue_planner_avoids_dangling_linguistic_boundaries() -> None:
    words = [
        {"text": "Transformers", "start_ms": 0, "end_ms": 300},
        {"text": "typically", "start_ms": 400, "end_ms": 700},
        {"text": "also", "start_ms": 800, "end_ms": 1100},
        {"text": "include", "start_ms": 1200, "end_ms": 1500},
        {"text": "a", "start_ms": 1600, "end_ms": 1900},
        {"text": "second", "start_ms": 2000, "end_ms": 2300},
        {"text": "type", "start_ms": 2400, "end_ms": 2700},
        {"text": "of", "start_ms": 2800, "end_ms": 3100},
        {"text": "operation", "start_ms": 3200, "end_ms": 3500},
        {"text": "known", "start_ms": 3600, "end_ms": 3900},
        {"text": "as", "start_ms": 4000, "end_ms": 4300},
        {"text": "attention.", "start_ms": 4400, "end_ms": 4700},
        {"text": "This", "start_ms": 5200, "end_ms": 5500},
        {"text": "matters.", "start_ms": 5600, "end_ms": 5900},
    ]

    cues = build_dubbing_cues(
        [{
            "segment_id": "seg_transformer",
            "start_ms": 0,
            "end_ms": words[-1]["end_ms"],
            "duration_ms": words[-1]["end_ms"],
            "source_text": " ".join(word["text"] for word in words),
            "words": words,
        }],
        target_duration_ms=4000,
        min_duration_ms=1500,
        max_duration_ms=4200,
    )

    joined = " | ".join(str(cue["source_text"]) for cue in cues)
    assert "known | as" not in joined
    assert "as | attention" not in joined
    assert "known as | attention" not in joined
    assert "type of | operation" not in joined
    assert any("operation known as attention." in str(cue["source_text"]) for cue in cues)


def test_cue_planner_splits_long_sentence_at_clause_not_dependency_boundary() -> None:
    text = (
        "An algorithm called back propagation is used to tweak all of the parameters, "
        "making the model a little more likely to choose the true last word."
    )
    words = [
        {"text": token, "start_ms": index * 320, "end_ms": index * 320 + 240}
        for index, token in enumerate(text.split())
    ]

    cues = build_dubbing_cues(
        [{
            "segment_id": "sentence_000001",
            "parent_segment_ids": ["seg_000001"],
            "source_text": text,
            "words": words,
            "start_ms": 0,
            "end_ms": words[-1]["end_ms"],
            "duration_ms": words[-1]["end_ms"],
        }],
        target_duration_ms=2800,
        min_duration_ms=1500,
        max_duration_ms=3600,
    )

    joined = " | ".join(str(cue["source_text"]) for cue in cues)
    assert "all | of" not in joined
    assert "likely | to" not in joined
    assert any(str(cue["source_text"]).endswith("parameters,") for cue in cues[:-1])
