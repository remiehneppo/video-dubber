from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from dubber.domain.profiles import (
    detect_protected_spans,
    glossary_terms_from_spans,
    load_domain_profile,
    normalize_spoken_text,
    protected_translation_errors,
)


def test_calculus_profile_detects_and_normalizes_core_notation() -> None:
    profile = load_domain_profile("mathematics", explicit_profile="calculus")
    text = "The derivative dy/dx and area πr² use dr and ½."

    spans = detect_protected_spans(text, profile)
    canonical = [span.canonical for span in spans]

    assert "dy/dx" in canonical
    assert "πr²" in canonical
    assert "dr" in canonical
    assert "½" in canonical
    assert normalize_spoken_text("dy/dx, πr², dr, ½", spans) == "d y trên d x, pi r bình phương, d r, một phần hai"


def test_calculus_spans_seed_locked_glossary_terms() -> None:
    profile = load_domain_profile("mathematics", explicit_profile="calculus")
    spans = detect_protected_spans("2 pi r times dr", profile)

    terms = glossary_terms_from_spans(spans, source_segments=["seg_1"], locked=True)

    dr = next(term for term in terms if term["original"] == "dr")
    assert dr["locked"] is True
    assert dr["vietnamese"] == "d r"
    assert "doctor" in dr["forbidden"]


def test_calculus_profile_protects_dx_squared_as_single_notation() -> None:
    profile = load_domain_profile("mathematics", explicit_profile="calculus")
    source_text = "plus whatever the change to x squared is, dx squared."

    spans = detect_protected_spans(source_text, profile)

    assert [span.canonical for span in spans] == ["x²", "dx²"]
    assert normalize_spoken_text("dx²", spans) == "d x bình phương"
    assert protected_translation_errors(source_text, "cộng với thay đổi của x bình phương", spans) == [
        "protected span dx² must be represented as d x bình phương"
    ]
    assert protected_translation_errors(source_text, "cộng với thay đổi dx²", spans) == []
    assert protected_translation_errors(source_text, "cộng với thay đổi d x bình phương", spans) == []


def test_calculus_profile_treats_x_of_t_squared_as_function_squared() -> None:
    profile = load_domain_profile("mathematics", explicit_profile="calculus")
    source_text = "Well, the derivative of x of t squared is 2 times x of t times the derivative of x."

    spans = detect_protected_spans(source_text, profile)

    assert [span.canonical for span in spans] == ["x(t)²"]
    assert protected_translation_errors(source_text, "Đạo hàm của x(t) bình phương bằng 2 nhân x(t).", spans) == []


def test_calculus_profile_loads_without_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def block_yaml(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_yaml)

    profile = load_domain_profile("mathematics", explicit_profile="calculus")

    assert profile.profile_id == "calculus"
    assert any(term["original"] == "dr" for term in profile.glossary_seeds)
    assert profile.forbidden_translations["dr"] == ["doctor", "bác sĩ", "tiến sĩ"]


def test_batch_calculus_fixture_rejects_doctor_for_dr() -> None:
    fixture = json.loads(Path("tests/fixtures/calculus/protected_spans.json").read_text(encoding="utf-8"))
    profile = load_domain_profile("mathematics", explicit_profile="calculus")

    for example in fixture["examples"]:
        spans = detect_protected_spans(example["source_text"], profile)
        assert spans, example["segment_id"]
        assert protected_translation_errors(example["source_text"], example["bad_vi_text"], spans)
        assert protected_translation_errors(example["source_text"], example["good_vi_text"], spans) == []
