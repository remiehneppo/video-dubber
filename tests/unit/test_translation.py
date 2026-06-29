from __future__ import annotations

import pytest

from dubber.translation.block_builder import build_translation_blocks, build_translation_context_blocks
from dubber.translation.validator import TranslationValidationError, validate_translations


def test_build_translation_blocks_uses_overlap_without_losing_segments() -> None:
    segments = [
        {"segment_id": f"seg_{index:06d}", "source_text": f"text {index}"}
        for index in range(1, 8)
    ]

    blocks = build_translation_blocks(segments, block_size=3, overlap=1)

    assert [[segment["segment_id"] for segment in block] for block in blocks] == [
        ["seg_000001", "seg_000002", "seg_000003"],
        ["seg_000003", "seg_000004", "seg_000005"],
        ["seg_000005", "seg_000006", "seg_000007"],
    ]


def test_build_translation_blocks_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        build_translation_blocks([{"segment_id": "seg_000001"}], block_size=3, overlap=3)


def test_build_translation_context_blocks_uses_word_windows_with_context() -> None:
    segments = [
        {"segment_id": f"seg_{index:06d}", "source_text": "word " * words}
        for index, words in enumerate([30, 35, 40, 45, 50, 55, 60], start=1)
    ]

    blocks = build_translation_context_blocks(
        segments,
        min_context_words=100,
        max_context_words=150,
        context_overlap_words=40,
        target_segment_count=3,
    )

    assert [[segment["segment_id"] for segment in block.target_segments] for block in blocks] == [
        ["seg_000001", "seg_000002", "seg_000003"],
        ["seg_000004", "seg_000005", "seg_000006"],
        ["seg_000007"],
    ]
    assert [segment["segment_id"] for segment in blocks[1].context_before] == ["seg_000003"]
    assert [segment["segment_id"] for segment in blocks[1].context_after] == ["seg_000007"]


def test_build_translation_context_blocks_allows_single_oversized_segment() -> None:
    segments = [
        {"segment_id": "seg_000001", "source_text": "word " * 500},
        {"segment_id": "seg_000002", "source_text": "short text"},
    ]

    blocks = build_translation_context_blocks(
        segments,
        min_context_words=120,
        max_context_words=350,
        context_overlap_words=40,
        target_segment_count=6,
    )

    assert [[segment["segment_id"] for segment in block.target_segments] for block in blocks] == [
        ["seg_000001"],
        ["seg_000002"],
    ]


def test_validate_translations_accepts_matching_segments_and_glossary() -> None:
    source_segments = [
        {
            "segment_id": "seg_000001",
            "source_text": "Let's talk about eigenvectors.",
        }
    ]
    translated_segments = [
        {
            "segment_id": "seg_000001",
            "vi_text": "Hãy nói về vectơ riêng.",
        }
    ]
    glossary_terms = [
        {
            "original": "eigenvectors",
            "vietnamese": "vectơ riêng",
            "locked": True,
        }
    ]

    report = validate_translations(
        source_segments,
        translated_segments,
        glossary_terms,
        max_length_ratio=2.0,
    )

    assert report.warning_count == 0


def test_validate_translations_rejects_missing_segment() -> None:
    with pytest.raises(TranslationValidationError, match="segment ids do not match"):
        validate_translations(
            [{"segment_id": "seg_000001", "source_text": "hello"}],
            [],
            [],
        )


def test_validate_translations_rejects_glossary_violation() -> None:
    with pytest.raises(TranslationValidationError, match="locked glossary term"):
        validate_translations(
            [{"segment_id": "seg_000001", "source_text": "eigenvector"}],
            [{"segment_id": "seg_000001", "vi_text": "một hướng đặc biệt"}],
            [{"original": "eigenvector", "vietnamese": "vectơ riêng", "locked": True}],
        )


def test_validate_translations_warns_when_translation_is_too_long() -> None:
    report = validate_translations(
        [{"segment_id": "seg_000001", "source_text": "short"}],
        [
            {
                "segment_id": "seg_000001",
                "vi_text": "đây là một câu dịch dài hơn nhiều so với bản gốc",
            }
        ],
        [],
        max_length_ratio=1.2,
    )

    assert report.warning_count == 1
    assert report.warnings[0]["warning"] == "length_ratio_exceeded"


def test_validate_translations_allows_empty_source_segments() -> None:
    report = validate_translations(
        [{"segment_id": "seg_000001", "source_text": "   "}],
        [{"segment_id": "seg_000001", "vi_text": ""}],
        [],
    )

    assert report.warning_count == 0
