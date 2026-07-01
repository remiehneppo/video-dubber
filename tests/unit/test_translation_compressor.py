from __future__ import annotations

from dubber.translation.compressor import compress_segment_translation


def test_compress_segment_translation_reduces_overlong_text_and_preserves_glossary() -> None:
    result = compress_segment_translation(
        {
            "segment_id": "seg_000001",
            "source_text": "eigenvector",
            "vi_text": "Bây giờ chúng ta hãy cùng nhau nói thật chi tiết về vectơ riêng trong ví dụ này.",
        },
        [{"original": "eigenvector", "vietnamese": "vectơ riêng", "locked": True}],
        max_length_ratio=3.0,
    )

    assert result.segment_id == "seg_000001"
    assert "vectơ riêng" in result.vi_text
    assert len(result.vi_text) < len("Bây giờ chúng ta hãy cùng nhau nói thật chi tiết về vectơ riêng trong ví dụ này.")
    assert "compressed_for_length" in result.warnings


def test_compress_segment_translation_leaves_short_text_unchanged() -> None:
    result = compress_segment_translation(
        {
            "segment_id": "seg_000001",
            "source_text": "eigenvector",
            "vi_text": "vectơ riêng",
        },
        [{"original": "eigenvector", "vietnamese": "vectơ riêng", "locked": True}],
        max_length_ratio=3.0,
    )

    assert result.vi_text == "vectơ riêng"
    assert result.warnings == []


def test_compress_segment_translation_does_not_prepend_missing_locked_term() -> None:
    result = compress_segment_translation(
        {
            "segment_id": "seg_000001",
            "source_text": "eigenvector",
            "vi_text": "một hướng đặc biệt dài dòng không đúng thuật ngữ",
        },
        [{"original": "eigenvector", "vietnamese": "vectơ riêng", "locked": True}],
        max_length_ratio=2.0,
    )

    assert not result.vi_text.startswith("vectơ riêng:")
    assert "vectơ riêng" not in result.vi_text
    assert "locked_glossary_missing" in result.warnings


def test_compress_segment_translation_never_hard_truncates_technical_meaning() -> None:
    technical_text = "Đạo hàm biểu diễn tốc độ thay đổi tức thời của hàm số theo biến x."

    result = compress_segment_translation(
        {
            "segment_id": "seg_technical",
            "source_text": "derivative",
            "vi_text": technical_text,
        },
        [{"original": "derivative", "vietnamese": "đạo hàm", "locked": True}],
        max_length_ratio=1.0,
    )

    assert result.vi_text == technical_text
    assert "length_compression_required" in result.warnings
