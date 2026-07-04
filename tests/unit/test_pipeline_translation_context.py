from __future__ import annotations

import json
from pathlib import Path

import pytest

from dubber.core.models import DubberConfig, DubbingCueConfig, ProjectConfig, TTSServiceConfig
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import _fallback_glossary_terms_from_text, _normalize_glossary_term, _review_required_items, run_glossary, run_translation
from dubber.providers.factory import ProviderBundle
from dubber.transcript.cues import build_dubbing_cues


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        assert payload["domain"] == "seo"
        if system_prompt.startswith("You extract"):
            segment_ids = [segment["segment_id"] for segment in payload["segments"]]
            record = {
                "system_prompt": system_prompt,
                "domain": payload["domain"],
                "segment_count": len(payload["segments"]),
                "segment_ids": segment_ids,
            }
            self.calls.append(record)
            return {
                "terms": [
                    {
                        "term_id": "term_ai",
                        "original": "AI",
                        "vietnamese": "trí tuệ nhân tạo",
                        "category": "topic",
                        "confidence": 0.98,
                        "source_segments": segment_ids,
                        "notes": "",
                    }
                ]
            }
        target_ids = [segment["segment_id"] for segment in payload["target_segments"]]
        context_ids = [segment["segment_id"] for segment in payload["context_before"] + payload["context_after"]]
        assert "Translate only target_segments" in system_prompt
        assert "context_before and context_after only" in system_prompt
        assert "Do not output translations for context-only segments" in system_prompt
        assert "mid-sentence" in system_prompt
        assert "Do not translate word by word" in system_prompt
        assert "rephrase the idea" in system_prompt
        assert "natural Vietnamese explanation" in system_prompt
        assert "no objects for context-only IDs" in payload["instruction"]
        record = {
            "system_prompt": system_prompt,
            "domain": payload["domain"],
            "segment_count": len(payload["target_segments"]),
            "segment_ids": target_ids,
            "context_ids": context_ids,
        }
        self.calls.append(record)
        context_echo = [
            {
                "segment_id": context_ids[0],
                "vi_text": "context-only",
                "used_terms": [],
                "length_ratio": 1.0,
                "translation_warnings": [],
            }
        ] if context_ids else []
        return {
            "segments": context_echo + [
                {
                    "segment_id": segment["segment_id"],
                    "vi_text": f"seo::{segment['segment_id']}",
                    "used_terms": ["AI"],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment in payload["target_segments"]
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
    translated = json.loads((paths.artifact_path("translated.v2.json")).read_text(encoding="utf-8"))

    assert [call["segment_count"] for call in llm.calls[:2]] == [9, 9]
    assert [call["segment_count"] for call in llm.calls[2:]] == [2]
    assert all(call["domain"] == "seo" for call in llm.calls)
    assert glossary["domain"] == "seo"
    assert len(glossary["terms"]) == 1
    assert len(translated["segments"]) == 9
    cues = json.loads(paths.artifact_path("dubbing_cues.v2.json").read_text(encoding="utf-8"))["cues"]
    assert all(cue["display_text"] == f"seo::{cue['cue_id']}" for cue in cues)
    assert translated["segments"][0]["display_text"].startswith("seo::cue_")


class OmitsSecondBlockOnceLLMProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        target_segments = payload["target_segments"]
        target_ids = [segment["segment_id"] for segment in target_segments]
        self.calls.append(target_ids)
        if len(self.calls) == 1:
            return {"segments": []}
        return {
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "vi_text": f"retry::{segment['segment_id']}",
                    "used_terms": [],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment in target_segments
            ]
        }


def test_translation_retries_block_when_target_segments_are_missing(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_retry_missing")
    segments = [
        {"segment_id": f"seg_{index:06d}", "start_ms": (index - 1) * 1000, "end_ms": index * 1000}
        for index in range(1, 10)
    ]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "seo", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_retry_missing", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_retry_missing", paths.manifest_file)
    llm = OmitsSecondBlockOnceLLMProvider()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=llm, tts=object()),
        config=DubberConfig(project=ProjectConfig(domain="seo"), dubbing_cues=DubbingCueConfig(1000, 1000, 1000)),
    )

    run_translation(ctx)

    translated = json.loads((paths.artifact_path("translated.v2.json")).read_text(encoding="utf-8"))
    assert any(llm.calls.count(target_ids) >= 2 for target_ids in llm.calls)
    assert [segment["segment_id"] for segment in translated["segments"]] == [segment["segment_id"] for segment in segments]
    assert translated["segments"][-1]["display_text"].startswith("retry::cue_")



class PartialTranslationLLMProvider:
    def __init__(self) -> None:
        self.target_ids_by_call: list[list[str]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        target_ids = [str(item["segment_id"]) for item in payload["target_segments"]]
        self.target_ids_by_call.append(target_ids)
        returned = target_ids[:2] if len(self.target_ids_by_call) == 1 else target_ids
        return {
            "segments": [
                {
                    "segment_id": segment_id,
                    "vi_text": f"partial::{segment_id}",
                    "used_terms": [],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment_id in returned
            ]
        }


def test_translation_retries_only_missing_segments_and_merges_partial_result(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_partial_retry")
    segments = [
        {"segment_id": f"seg_{index:06d}", "start_ms": (index - 1) * 1000, "end_ms": index * 1000}
        for index in range(1, 5)
    ]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "seo", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_partial_retry", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_partial_retry", paths.manifest_file)
    llm = PartialTranslationLLMProvider()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=llm, tts=object()),
        config=DubberConfig(project=ProjectConfig(domain="seo"), dubbing_cues=DubbingCueConfig(1000, 1000, 1000)),
    )

    run_translation(ctx)

    assert len(llm.target_ids_by_call[0]) == 4
    assert llm.target_ids_by_call[1] == llm.target_ids_by_call[0][2:]
    translated = json.loads(paths.artifact_path("translated.v2.json").read_text(encoding="utf-8"))
    assert [item["segment_id"] for item in translated["segments"]] == [item["segment_id"] for item in segments]


class BoundaryDedupeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        if system_prompt.startswith("You extract"):
            return {"terms": []}
        target_segments = payload["target_segments"]
        target_ids = [segment["segment_id"] for segment in target_segments]
        self.calls.append(target_ids)
        return {
            "segments": [
                {
                    "segment_id": target_segments[0]["segment_id"],
                    "vi_text": "Cụm mở đầu. Phần tiếp theo.",
                    "used_terms": [],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                },
                {
                    "segment_id": target_segments[1]["segment_id"],
                    "vi_text": "Cụm mở đầu. Phần tiếp theo mở rộng.",
                    "used_terms": [],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                },
            ]
        }


def test_translation_trims_repeated_boundary_prefix_before_cue_assembly(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_boundary_dedupe")
    segments = [
        {"segment_id": "seg_000001", "start_ms": 0, "end_ms": 1000, "source_text": "First source sentence.", "source_text_raw": "First source sentence."},
        {"segment_id": "seg_000002", "start_ms": 1000, "end_ms": 2000, "source_text": "Second source sentence.", "source_text_raw": "Second source sentence."},
    ]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "seo", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_boundary_dedupe", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_boundary_dedupe", paths.manifest_file)
    llm = BoundaryDedupeLLMProvider()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=llm, tts=object()),
        config=DubberConfig(project=ProjectConfig(domain="seo"), dubbing_cues=DubbingCueConfig(1000, 1000, 1000)),
    )

    run_translation(ctx)

    translated = json.loads(paths.artifact_path("translated.v2.json").read_text(encoding="utf-8"))
    cues = json.loads(paths.artifact_path("dubbing_cues.v2.json").read_text(encoding="utf-8"))["cues"]
    assert translated["segments"][0]["display_text"] == "Cụm mở đầu. Phần tiếp theo."
    assert translated["segments"][1]["display_text"] == "mở rộng."
    assert cues[0]["display_text"] == "Cụm mở đầu. Phần tiếp theo."
    assert cues[1]["display_text"] == "mở rộng."
    assert "boundary_overlap_trimmed" in cues[1]["translation_warnings"]


class ProtectedSpanRetryLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        self.calls.append({"system_prompt": system_prompt, "payload": payload})
        segment_id = payload["target_segments"][0]["segment_id"]
        assert payload["domain_profile"] == "calculus@1"
        assert payload["protected_spans"][segment_id][0]["canonical"] == "dr"
        if len(self.calls) == 1:
            return {
                "segments": [
                    {
                        "segment_id": segment_id,
                        "vi_text": "Diện tích là 2 pi r nhân doctor.",
                        "used_terms": [],
                        "length_ratio": 1.0,
                        "translation_warnings": [],
                    }
                ]
            }
        assert "Protected notation correction required" in system_prompt
        return {
            "segments": [
                {
                    "segment_id": segment_id,
                    "vi_text": "Diện tích là 2 pi r nhân d r.",
                    "spoken_text": "Diện tích là 2 pi r nhân doctor.",
                    "used_terms": ["dr"],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
            ]
        }


def test_translation_retries_once_for_calculus_protected_span_violation(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_protected_retry")
    segments = [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 4000}]
    _write_segments(paths, segments)
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 4000,
                        "duration_ms": 4000,
                        "source_text": "Its area is 2 pi r times dr.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "domain": "mathematics",
                "domain_profile": "calculus@1",
                "status": "locked",
                "terms": [
                    {
                        "original": "dr",
                        "vietnamese": "d r",
                        "locked": True,
                        "protected": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_protected_retry", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_protected_retry", paths.manifest_file)
    llm = ProtectedSpanRetryLLMProvider()
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=llm, tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(4000, 1000, 6000),
        ),
    )

    run_translation(ctx)

    assert len(llm.calls) == 2
    cues = json.loads(paths.artifact_path("dubbing_cues.v2.json").read_text(encoding="utf-8"))["cues"]
    assert cues[0]["display_text"] == "Diện tích là 2 pi r nhân d r."
    assert cues[0]["spoken_text"] == "Diện tích là 2 pi r nhân d r."
    assert "doctor" not in cues[0]["display_text"].lower()
    translated = json.loads(paths.artifact_path("translated.v2.json").read_text(encoding="utf-8"))
    translated_artifact = json.loads(paths.artifact_path("translated.v2.json").read_text(encoding="utf-8"))
    assert translated["domain_profile"] == "calculus@1"
    assert translated["segments"][0]["protected_spans"][0]["canonical"] == "dr"
    assert translated_artifact["schema_version"] == "2.0"
    assert translated_artifact["segments"][0]["display_text"] == "Diện tích là 2 pi r nhân d r."


class ReviewCueLLMProvider:
    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        return {
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "vi_text": "Bản dịch trước review d r.",
                    "used_terms": ["dr"],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment in payload["target_segments"]
            ]
        }


class DenseTranslationLLMProvider(ReviewCueLLMProvider):
    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        payload = json.loads(user_prompt)
        return {
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "vi_text": "Nội dung kỹ thuật này quá dài để đọc trong cửa sổ ngắn.",
                    "used_terms": [],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment in payload["target_segments"]
            ]
        }


def test_review_required_items_are_chronological_and_include_timing() -> None:
    cues = [
        {
            "cue_id": "cue_late",
            "start_ms": 3000,
            "end_ms": 4000,
            "duration_ms": 1000,
            "source_text_raw": "late raw",
            "source_text": "late normalized",
            "display_text": "late display",
            "spoken_text": "late spoken",
            "normalization_edits": [{"rule_id": "late"}],
        },
        {
            "cue_id": "cue_early",
            "start_ms": 1000,
            "end_ms": 1800,
            "duration_ms": 800,
            "source_text_raw": "early raw",
            "source_text": "early normalized",
            "display_text": "early display",
            "spoken_text": "early spoken",
            "risk_flags": ["tts_timing_density_high"],
        },
    ]

    items = _review_required_items(cues)

    assert [item["cue_id"] for item in items] == ["cue_early", "cue_late"]
    assert [item["start_ms"] for item in items] == [1000, 3000]
    assert items[0]["review_overrides"]["start_ms"] == 1000
    assert items[0]["review_overrides"]["end_ms"] == 1800


def test_translation_review_required_pauses_and_locked_review_overrides_text(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_review_required")
    _write_segments(paths, [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 4000}])
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 4000,
                        "duration_ms": 4000,
                        "source_text_raw": "The thickness is doctor.",
                        "source_text": "The thickness is dr.",
                        "source_text_normalized": "The thickness is dr.",
                        "normalization_confidence": 0.92,
                        "normalization_edits": [
                            {
                                "rule_id": "calculus.doctor_to_dr",
                                "original": "doctor",
                                "normalized": "dr",
                                "confidence": 0.92,
                            }
                        ],
                        "risk_flags": ["source_normalized_calculus_notation"],
                        "words": [
                            {"text": "The", "start_ms": 0, "end_ms": 200},
                            {"text": "thickness", "start_ms": 300, "end_ms": 800},
                            {"text": "is", "start_ms": 900, "end_ms": 1000},
                            {"text": "dr.", "start_ms": 1100, "end_ms": 1300},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "mathematics", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_review_required", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_review_required", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(4000, 1000, 6000),
        ),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    assert required["status"] == "required"
    assert required["cues"][0]["reason"] == "source_normalization_review_required"
    assert required["cues"][0]["source_text_raw"] == "The thickness is doctor."

    required["status"] = "locked"
    required["cues"][0]["review_overrides"]["display_text"] = "Bản dịch đã review d r."
    required["cues"][0]["review_overrides"]["spoken_text"] = "Bản dịch đã review d r."
    paths.artifact_path("review.locked.json").write_text(json.dumps(required, ensure_ascii=False), encoding="utf-8")

    assert run_translation(ctx) is False
    cues_v2 = json.loads(paths.artifact_path("dubbing_cues.v2.json").read_text(encoding="utf-8"))["cues"]
    assert cues_v2[0]["display_text"] == "Bản dịch đã review d r."
    assert cues_v2[0]["spoken_text"] == "Bản dịch đã review d r."


def test_translation_locked_review_applies_safe_timeline_override(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_review_timeline")
    _write_segments(paths, [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 4000}])
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 4000,
                        "duration_ms": 4000,
                        "source_text_raw": "The thickness is doctor.",
                        "source_text": "The thickness is dr.",
                        "source_text_normalized": "The thickness is dr.",
                        "normalization_confidence": 0.92,
                        "normalization_edits": [{"rule_id": "calculus.doctor_to_dr"}],
                        "risk_flags": ["source_normalized_calculus_notation"],
                        "words": [
                            {"text": "The", "start_ms": 0, "end_ms": 200},
                            {"text": "thickness", "start_ms": 300, "end_ms": 800},
                            {"text": "is", "start_ms": 900, "end_ms": 1000},
                            {"text": "dr.", "start_ms": 1100, "end_ms": 1300},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "mathematics", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_review_timeline", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_review_timeline", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(4000, 1000, 6000),
        ),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    required["status"] = "locked"
    required["cues"][0]["review_overrides"]["start_ms"] = 120
    required["cues"][0]["review_overrides"]["end_ms"] = 3880
    paths.artifact_path("review.locked.json").write_text(json.dumps(required, ensure_ascii=False), encoding="utf-8")

    assert run_translation(ctx) is False
    cues_v2 = json.loads(paths.artifact_path("dubbing_cues.v2.json").read_text(encoding="utf-8"))["cues"]
    assert cues_v2[0]["start_ms"] == 120
    assert cues_v2[0]["end_ms"] == 3880
    assert cues_v2[0]["duration_ms"] == 3760
    timeline = json.loads(paths.artifact_path("speech_timeline.v1.json").read_text(encoding="utf-8"))
    assert timeline["speech_intervals"][0]["start_ms"] == 120


def test_translation_locked_review_rejects_overlapping_timeline_override(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_review_timeline_overlap")
    segments = [
        {"segment_id": "seg_000001", "start_ms": 0, "end_ms": 2000},
        {"segment_id": "seg_000002", "start_ms": 2500, "end_ms": 4500},
    ]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "mathematics", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    cue_id = str(build_dubbing_cues(
        [
            {
                **segment,
                "source_text": f"source {segment['segment_id']}",
            }
            for segment in segments
        ],
        target_duration_ms=2000,
        min_duration_ms=1000,
        max_duration_ms=2200,
    )[0]["cue_id"])
    locked = {
        "schema_version": "1.0",
        "status": "locked",
        "cues": [
            {
                "cue_id": cue_id,
                "review_overrides": {
                    "start_ms": 0,
                    "end_ms": 3000,
                },
            }
        ],
    }
    paths.artifact_path("review.locked.json").write_text(json.dumps(locked), encoding="utf-8")
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_review_timeline_overlap",
        input_file=Path("input/video.mp4"),
    )
    manifest = ArtifactManifest.create("job_review_timeline_overlap", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(2000, 1000, 2200),
        ),
    )

    with pytest.raises(ValueError, match="overlaps next cue"):
        run_translation(ctx)


def test_translation_stale_review_lock_does_not_approve_new_high_risk_cue(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_stale_review_lock")
    _write_segments(paths, [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 2000}])
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 2000,
                        "duration_ms": 2000,
                        "source_text_raw": "Repeated raw text.",
                        "source_text": "Normalized text.",
                        "normalization_edits": [
                            {
                                "rule_id": "generic.repeated_asr_loop",
                                "original": "Repeated raw text.",
                                "normalized": "Normalized text.",
                            }
                        ],
                        "risk_flags": ["source_normalized_repetitive_asr_loop"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "AI", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    paths.artifact_path("review.locked.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "locked",
                "cues": [{"cue_id": "cue_from_old_transcript", "review_overrides": {"spoken_text": "old"}}],
            }
        ),
        encoding="utf-8",
    )
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_stale_review_lock",
        input_file=Path("input/video.mp4"),
    )
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=ArtifactManifest.create("job_stale_review_lock", paths.manifest_file),
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(project=ProjectConfig(domain="AI")),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    assert required["cues"][0]["cue_id"] != "cue_from_old_transcript"


def test_translation_requires_review_when_spoken_text_cannot_fit_timing_budget(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_tts_timing_review")
    segments = [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 1200}]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "AI", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    ctx = StageContext(
        paths=paths,
        store=CheckpointStore.create(
            paths.job_state_file,
            job_id="job_tts_timing_review",
            input_file=Path("input/video.mp4"),
        ),
        manifest=ArtifactManifest.create("job_tts_timing_review", paths.manifest_file),
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=DenseTranslationLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="AI"),
            dubbing_cues=DubbingCueConfig(1200, 1000, 1600),
            tts_service=TTSServiceConfig(max_speedup_ratio=1.2),
        ),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    assert "tts_timing_density_high" in required["cues"][0]["risk_flags"]


def test_translation_review_required_for_llm_source_normalization_suggestion(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_review_suggestion")
    _write_segments(paths, [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 4000}])
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 4000,
                        "duration_ms": 4000,
                        "source_text_raw": "The doctor appears in notation.",
                        "source_text": "The doctor appears in notation.",
                        "source_text_normalized": "The doctor appears in notation.",
                        "normalization_confidence": 1.0,
                        "normalization_edits": [],
                        "normalization_suggestions": [
                            {
                                "candidate_id": "seg_000001:doctor:1",
                                "original": "doctor",
                                "suggested_normalized": "dr",
                                "confidence": 0.84,
                                "reason": "May be calculus notation.",
                                "rule_id": "llm.source_normalization_suggestion",
                            }
                        ],
                        "risk_flags": ["source_normalization_llm_suggestion"],
                        "words": [
                            {"text": "The", "start_ms": 0, "end_ms": 200},
                            {"text": "doctor", "start_ms": 300, "end_ms": 700},
                            {"text": "appears", "start_ms": 800, "end_ms": 1100},
                            {"text": "in", "start_ms": 1200, "end_ms": 1300},
                            {"text": "notation.", "start_ms": 1400, "end_ms": 1800},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "mathematics", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_review_suggestion", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_review_suggestion", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(4000, 1000, 6000),
        ),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    assert required["cues"][0]["normalization_edits"] == []
    assert required["cues"][0]["normalization_suggestions"][0]["suggested_normalized"] == "dr"


def test_translation_requires_review_for_high_risk_vad_boundary(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_review_vad_boundary")
    segments = [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 4000}]
    _write_segments(paths, segments)
    _write_transcript(paths, segments)
    transcript_path = paths.artifact_path("transcript.v1.json")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript["segments"][0].update({
        "source_text_raw": "source seg_000001",
        "risk_flags": ["hard_split"],
        "vad_split_reason": "vad_hard_split",
        "vad_risk_flags": ["hard_split"],
        "words": [
            {"text": "source", "start_ms": 0, "end_ms": 1000},
            {"text": "seg_000001", "start_ms": 1200, "end_ms": 3000},
        ],
    })
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "mathematics", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_review_vad_boundary",
        input_file=Path("input/video.mp4"),
    )
    manifest = ArtifactManifest.create("job_review_vad_boundary", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(4000, 1000, 6000),
        ),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    assert required["cues"][0]["reason"] == "asr_timeline_review_required"
    assert required["cues"][0]["risk_flags"] == ["hard_split"]


def test_translation_attaches_normalization_review_only_to_affected_child_cue(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_review_affected_cue")
    _write_segments(paths, [{"segment_id": "seg_000001", "start_ms": 0, "end_ms": 3500}])
    paths.artifact_path("transcript.v1.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "segments": [{
                "segment_id": "seg_000001",
                "start_ms": 0,
                "end_ms": 3500,
                "duration_ms": 3500,
                "source_text_raw": "Intro words. thickness doctor.",
                "source_text": "Intro words. thickness dr.",
                "source_text_normalized": "Intro words. thickness dr.",
                "normalization_confidence": 0.92,
                "normalization_edits": [{
                    "rule_id": "calculus.doctor_to_dr",
                    "original": "doctor",
                    "normalized": "dr",
                    "start_char": 23,
                    "end_char": 29,
                    "confidence": 0.92,
                    "reason": "calculus context",
                }],
                "risk_flags": ["source_normalized_calculus_notation"],
                "words": [
                    {"text": "Intro", "raw_text": "Intro", "start_ms": 0, "end_ms": 400},
                    {"text": "words.", "raw_text": "words.", "start_ms": 500, "end_ms": 1000},
                    {"text": "thickness", "raw_text": "thickness", "start_ms": 2500, "end_ms": 3000},
                    {"text": "dr.", "raw_text": "doctor.", "start_ms": 3100, "end_ms": 3500},
                ],
            }],
        }),
        encoding="utf-8",
    )
    paths.artifact_path("glossary.locked.json").write_text(
        json.dumps({"schema_version": "1.0", "domain": "mathematics", "status": "locked", "terms": []}),
        encoding="utf-8",
    )
    store = CheckpointStore.create(
        paths.job_state_file,
        job_id="job_review_affected_cue",
        input_file=Path("input/video.mp4"),
    )
    manifest = ArtifactManifest.create("job_review_affected_cue", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=object(), llm=ReviewCueLLMProvider(), tts=object()),
        config=DubberConfig(
            project=ProjectConfig(domain="mathematics", domain_profile="calculus"),
            dubbing_cues=DubbingCueConfig(1000, 500, 1800),
        ),
    )

    assert run_translation(ctx) is True
    required = json.loads(paths.artifact_path("review.required.json").read_text(encoding="utf-8"))
    assert len(required["cues"]) == 1
    assert required["cues"][0]["source_text_raw"] == "thickness doctor."


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
