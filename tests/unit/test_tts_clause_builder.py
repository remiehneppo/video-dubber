from __future__ import annotations

from dubber.tts.clause_builder import build_tts_clauses, build_tts_work_items


def test_build_tts_clauses_splits_matching_translation_at_source_pause() -> None:
    segment = {
        "segment_id": "seg_000026",
        "start_ms": 1000,
        "end_ms": 4500,
        "duration_ms": 3500,
        "source_text": "First sentence. Second sentence.",
        "timestamp_source": "word",
        "words": [
            {"text": "First", "start_ms": 1000, "end_ms": 1400},
            {"text": "sentence.", "start_ms": 1500, "end_ms": 2000},
            {"text": "Second", "start_ms": 3200, "end_ms": 3600},
            {"text": "sentence.", "start_ms": 3700, "end_ms": 4500},
        ],
    }

    clauses = build_tts_clauses(segment, "Cau thu nhat. Cau thu hai.", min_pause_ms=700)

    assert [clause.segment_id for clause in clauses] == [
        "seg_000026__clause_001",
        "seg_000026__clause_002",
    ]
    assert [(clause.start_ms, clause.end_ms) for clause in clauses] == [(1000, 2000), (3200, 4500)]
    assert [clause.translated_text for clause in clauses] == ["Cau thu nhat.", "Cau thu hai."]


def test_build_tts_clauses_falls_back_when_translation_sentence_count_does_not_match() -> None:
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 4500,
        "duration_ms": 3500,
        "source_text": "First. Second.",
        "timestamp_source": "word",
        "words": [
            {"text": "First.", "start_ms": 1000, "end_ms": 1800},
            {"text": "Second.", "start_ms": 3000, "end_ms": 4500},
        ],
    }

    clauses = build_tts_clauses(segment, "Mot cau dich duy nhat.", min_pause_ms=700)

    assert len(clauses) == 1
    assert clauses[0].segment_id == "seg_000001"
    assert clauses[0].translated_text == "Mot cau dich duy nhat."

def test_build_tts_work_items_merges_short_incomplete_fragment_with_following_segment() -> None:
    segments = [
        {
            "segment_id": "seg_000033",
            "start_ms": 366661,
            "end_ms": 367202,
            "duration_ms": 541,
            "source_text": "At the end,",
            "timestamp_source": "word",
            "words": [{"text": "end,", "start_ms": 367082, "end_ms": 367202}],
        },
        {
            "segment_id": "seg_000034",
            "start_ms": 367222,
            "end_ms": 381902,
            "duration_ms": 14680,
            "source_text": "the model predicts the next word.",
            "timestamp_source": "word",
            "words": [{"text": "the", "start_ms": 367222, "end_ms": 367400}],
        },
    ]
    translations = {
        "seg_000033": {"vi_text": "Cuoi cung,"},
        "seg_000034": {"vi_text": "mo hinh du doan tu tiep theo."},
    }

    items = build_tts_work_items(segments, translations)

    assert len(items) == 1
    assert items[0].parent_segment_ids == ("seg_000033", "seg_000034")
    assert items[0].segment["segment_id"] == "seg_000033__merged__seg_000034"
    assert items[0].segment["duration_ms"] == 15241
    assert items[0].translated_text == "Cuoi cung, mo hinh du doan tu tiep theo."


def test_build_tts_work_items_does_not_merge_completed_short_sentence() -> None:
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 1800,
        "duration_ms": 800,
        "source_text": "Done.",
    }
    following = {
        "segment_id": "seg_000002",
        "start_ms": 1810,
        "end_ms": 4000,
        "duration_ms": 2190,
        "source_text": "Next sentence.",
    }

    items = build_tts_work_items([segment, following], {})

    assert [item.parent_segment_ids for item in items] == [("seg_000001",), ("seg_000002",)]
