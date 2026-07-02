from __future__ import annotations

from dubber.tts.semantic import compare_tts_transcript, normalize_semantic_text


def test_semantic_comparison_normalizes_unicode_punctuation_and_spacing() -> None:
    assert normalize_semantic_text("  Xin CHÀO, thế giới! ") == "xin chào thế giới"
    metrics = compare_tts_transcript("Xin chào, thế giới!", "xin chào thế giới")
    assert metrics.cer == 0.0
    assert metrics.token_recall == 1.0
    assert metrics.ok(max_cer=0.25, min_token_recall=0.85)


def test_semantic_comparison_rejects_missing_or_wrong_speech() -> None:
    metrics = compare_tts_transcript("Đây là phép tính giải tích", "Đây là phép tính")
    assert metrics.token_recall < 0.85
    assert not metrics.ok(max_cer=0.25, min_token_recall=0.85)


def test_semantic_comparison_accepts_matching_empty_silence() -> None:
    metrics = compare_tts_transcript("", "")

    assert metrics.cer == 0.0
    assert metrics.token_recall == 1.0
    assert metrics.ok(max_cer=0.25, min_token_recall=0.85)


def test_semantic_comparison_accepts_common_math_asr_variants() -> None:
    assert normalize_semantic_text("2PR doctor") == "hai pi r dr"

    metrics = compare_tts_transcript(
        "phép xấp xỉ 2 pi r dr sẽ ngày càng ít",
        "Phép sắp xỉ 2PR doctor sẽ ngày càng ít",
    )

    assert metrics.ok(max_cer=0.25, min_token_recall=0.60)

def test_semantic_comparison_accepts_collapsed_two_pi_r_dr_less_variant() -> None:
    metrics = compare_tts_transcript(
        "phép xấp xỉ 2 pi r nhân d r ít",
        "Phép sắp xỉ 2PR x DRX",
    )

    assert metrics.expected_text == "phép xấp xỉ hai pi r nhân d r ít"
    assert metrics.transcript == "phép xấp xỉ hai pi r nhân d r ít"
    assert metrics.ok(max_cer=0.25, min_token_recall=0.85), metrics


def test_semantic_comparison_accepts_real_calculus_asr_variants() -> None:
    examples = [
        (
            "Nên diện tích của nó, bằng ½ × đáy × chiều cao, tính ra chính xác là",
            "nên diện tích của nó bằng 1 phần 2 nhân đáy nhân triều cao tính ra chính xác là",
        ),
        (
            "Sau 1 giây, ô tô đi được 1³ bằng 1 mét.",
            "Sau một giây, ô tô đi được một lập phương bằng một mét.",
        ),
        (
            "Mỗi hình vuông mỏng có thể tích x² nhân dx,",
            "Mỗi hình vuông mỏng có thể tích ích bình phương nhân BX.",
        ),
        (
            "trên giây, nên đạo hàm dy/dt đó bằng âm 1 mét trên giây.",
            "trên dây Nên đạo hàm di trên PT đó bằng âm 1m trên dây",
        ),
        (
            "0 chia 0 tại đầu vào, và tìm giới hạn khi x tiến tới đầu vào đó?",
            "Không chia không tại đầu vào và tìm giới hạn khi ít tiến tới đầu vào đó.",
        ),
        (
            "0, nó bằng âm 1.",
            "Không, nó bằng âm 1.",
        ),
    ]

    for expected, transcript in examples:
        metrics = compare_tts_transcript(expected, transcript)
        assert metrics.ok(max_cer=0.25, min_token_recall=0.60), metrics

