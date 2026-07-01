from __future__ import annotations

import json
from pathlib import Path


def test_glossary_and_translation_assets_exist_and_are_valid_json() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_paths = [
        root / "schemas" / "glossary.schema.json",
        root / "schemas" / "translated.schema.json",
        root / "schemas" / "translated.v2.schema.json",
        root / "schemas" / "dubbing_cues.v2.schema.json",
    ]

    for schema_path in schema_paths:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        assert payload["type"] == "object"
        assert payload["required"]


def test_pipeline_artifacts_with_public_contracts_have_schema_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_paths = [
        root / "schemas" / "source_normalization.v1.schema.json",
        root / "schemas" / "speech_timeline.v1.schema.json",
        root / "schemas" / "review.required.schema.json",
        root / "schemas" / "review.locked.schema.json",
    ]

    for schema_path in schema_paths:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["type"] == "object"
        assert payload["required"]


def test_prompt_templates_exist_with_required_placeholders() -> None:
    root = Path(__file__).resolve().parents[2]
    glossary_prompt = (root / "dubber" / "prompts" / "glossary_extract.md").read_text(encoding="utf-8")
    translate_prompt = (root / "dubber" / "prompts" / "translate_block.md").read_text(encoding="utf-8")
    compress_prompt = (root / "dubber" / "prompts" / "compress_translation.md").read_text(encoding="utf-8")

    assert "{domain}" in glossary_prompt
    assert "fewer high-value terms" in glossary_prompt
    assert "{glossary}" in translate_prompt
    assert "{segments}" in translate_prompt
    assert "context_before and context_after only" in translate_prompt
    assert "mid-sentence" in translate_prompt
    assert "length_ratio" in translate_prompt
    assert "translation_warnings" in translate_prompt
    assert "{segment}" in compress_prompt
    assert "Remove redundancy" in compress_prompt
