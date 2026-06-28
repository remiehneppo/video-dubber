from __future__ import annotations

import json
from pathlib import Path

from dubber.core.models import DubberConfig, ProjectConfig
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import _fallback_glossary_terms_from_text, _normalize_glossary_term, run_glossary, run_translation
from dubber.providers.factory import ProviderBundle


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        record = {
            "system_prompt": system_prompt,
            "domain": payload["domain"],
            "segment_count": len(payload["segments"]),
            "segment_ids": [segment["segment_id"] for segment in payload["segments"]],
        }
        self.calls.append(record)
        assert payload["domain"] == "seo"
        if system_prompt.startswith("Extract glossary"):
            return {
                "terms": [
                    {
                        "term_id": "term_ai",
                        "original": "AI",
                        "vietnamese": "trí tuệ nhân tạo",
                        "category": "topic",
                        "confidence": 0.98,
                        "source_segments": record["segment_ids"],
                        "notes": "",
                    }
                ]
            }
        return {
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "vi_text": f"seo::{segment['segment_id']}",
                    "used_terms": ["AI"],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment in payload["segments"]
            ]
        }


def test_fallback_glossary_terms_from_prose_bullets() -> None:
    terms = _fallback_glossary_terms_from_text(
        """
        Here is the glossary:
        * **Large Language Model (LLM)**: A model trained on text.
        * **Parameters (Weights)**: Learned numeric values.
        """,
        block_index=1,
        block_segment_ids=["seg_000001"],
        glossary_review=False,
    )

    assert [term["original"] for term in terms] == ["Large Language Model (LLM)", "Parameters (Weights)"]
    assert terms[0]["source_segments"] == ["seg_000001"]
    assert terms[0]["locked"] is True


def test_normalize_glossary_term_accepts_string_items() -> None:
    term = _normalize_glossary_term(
        "Large language models",
        block_index=1,
        term_index=2,
        block_segment_ids=["seg_000001", "seg_000002"],
        glossary_review=False,
    )

    assert term is not None
    assert term["term_id"] == "term_01_0002"
    assert term["original"] == "Large language models"
    assert term["source_segments"] == ["seg_000001", "seg_000002"]
    assert term["locked"] is True


def test_translation_uses_blocks_and_domain_context(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_domain")
    segments = [
        {"segment_id": f"seg_{index:06d}", "start_ms": (index - 1) * 1000, "end_ms": index * 1000}
        for index in range(1, 10)
    ]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    store = CheckpointStore.create(paths.job_state_file, job_id="job_domain", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_domain", paths.manifest_file)
    llm = FakeLLMProvider()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=llm, tts=object()),
        config=DubberConfig(project=ProjectConfig(domain="seo")),
    )

    assert run_glossary(ctx, glossary_review=False) is False
    run_translation(ctx)

    glossary = json.loads((paths.artifact_path("glossary.locked.json")).read_text(encoding="utf-8"))
    translated = json.loads((paths.artifact_path("translated.v1.json")).read_text(encoding="utf-8"))

    assert [call["segment_count"] for call in llm.calls[:2]] == [8, 2]
    assert [call["segment_count"] for call in llm.calls[2:]] == [6, 4]
    assert all(call["domain"] == "seo" for call in llm.calls)
    assert glossary["domain"] == "seo"
    assert len(glossary["terms"]) == 1
    assert len(translated["segments"]) == 9
    assert translated["segments"][0]["vi_text"] == "seo::seg_000001"
    assert translated["segments"][-1]["vi_text"] == "seo::seg_000009"


def _write_segments(paths: WorkspacePaths, segments: list[dict[str, object]]) -> None:
    paths.artifact_path("segments.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": paths.root.name,
                "source_audio": "audio/vocals.wav",
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "start_ms": segment["start_ms"],
                        "end_ms": segment["end_ms"],
                        "duration_ms": int(segment["end_ms"]) - int(segment["start_ms"]),
                    }
                    for segment in segments
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_transcript(paths: WorkspacePaths, segments: list[dict[str, object]]) -> None:
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provider": {"type": "openai_compatible", "model": "provider-asr"},
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "start_ms": segment["start_ms"],
                        "end_ms": segment["end_ms"],
                        "source_text": f"source {segment['segment_id']}",
                        "confidence": 0.9,
                        "asr_warnings": [],
                        "raw_response_path": f"raw/asr/{segment['segment_id']}.json",
                    }
                    for segment in segments
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
