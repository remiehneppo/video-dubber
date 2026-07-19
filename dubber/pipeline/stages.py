from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from dubber.asr.timestamps import NormalizedASRTimestamps, TimestampUnit, normalize_asr_timestamps
from dubber.asr.word_chunk_payloads import build_asr_word_chunk_payloads
from dubber.audio.vad import VadConfig as AudioVadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.concurrency import TaskCompletion, run_bounded
from dubber.core.io import read_json, write_json_atomic
from dubber.domain.profiles import (
    DomainProfile,
    ProtectedSpan,
    detect_protected_spans,
    glossary_terms_from_spans,
    load_domain_profile,
    normalize_spoken_text,
    protected_spans_prompt,
    protected_translation_errors,
)
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext
from dubber.providers.llm_openai_compatible import LLMStructuredOutputError
from dubber.subtitles.ass import build_spoken_subtitle_cues, render_ass
from dubber.tts.interactive_review import (
    TTSReviewDecision,
    TTSReviewRequest,
    is_manual_tts_review_error,
    load_tts_interactive_overrides,
    save_tts_interactive_override,
)
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_segment
from dubber.tts.track_mixer import assemble_commentary_track
from dubber.transcript.segmentation import build_transcript_segments
from dubber.transcript.cues import build_dubbing_cues
from dubber.transcript.normalization import (
    attach_source_normalization_suggestions,
    build_word_timeline,
    find_source_normalization_candidates,
    normalize_transcript_segments,
)
from dubber.transcript.sentences import build_source_sentences
from dubber.transcript.timeline import build_speech_timeline
from dubber.translation.block_builder import TranslationContextBlock, build_translation_context_blocks
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.glossary import normalize_glossary_terms
from dubber.translation.validator import validate_translations

logger = logging.getLogger(__name__)

_HIGH_RISK_CUE_FLAGS = frozenset({
    "chunk_text_fallback",
    "cue_duration_exceeds_max_for_safe_boundary",
    "hard_split",
    "no_speech_detected",
    "segment_over_hard_max",
    "segment_over_max_duration",
    "timestamps_missing",
    "source_normalized_repetitive_asr_loop",
    "tts_timing_density_high",
    "word_timestamps_missing",
})
_BOUNDARY_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TranslationProtectedSpanManualReviewRequired(ValueError):
    def __init__(self, *, segment_id: str, errors: list[str], review_item: dict[str, object]) -> None:
        super().__init__(f"protected span violation for {segment_id}: {'; '.join(errors)}")
        self.segment_id = segment_id
        self.errors = errors
        self.review_item = review_item


def _glossary_system_prompt(domain: str, profile: DomainProfile | None = None) -> str:
    profile_text = (
        f"Domain profile {profile.artifact_id}: {profile.prompt_summary}\n"
        if profile is not None and profile.profile_id != "generic"
        else ""
    )
    return (
        f"You extract a compact locked glossary for a {domain} educational video transcript block.\n"
        f"{profile_text}"
        "Return raw JSON only.\n"
        "Select only terms that materially affect translation consistency: domain terms, proper nouns, abbreviations, formulas, symbols, and repeated key phrases.\n"
        "Do not include generic filler words or ordinary phrases unless they have a domain-specific meaning.\n"
        "For Vietnamese, use concise natural terminology suitable for spoken educational commentary.\n"
        "Preserve names, formulas, symbols, units, and conventional notation exactly when translation would be harmful.\n"
        "Each term must cite source_segments from the provided segment_id values.\n"
        "Use confidence to reflect certainty; prefer fewer high-value terms over a long noisy glossary."
    )


def _glossary_instruction() -> str:
    return (
        "Return exactly one JSON object with a terms array. "
        "Each term should include term_id, original, vietnamese, category, confidence, source_segments, and notes. "
        "Keep vietnamese concise; leave notes empty unless a translation choice needs clarification."
    )


def _translation_system_prompt(domain: str, profile: DomainProfile | None = None, retry_instruction: str = "") -> str:
    profile_text = (
        f"Domain profile {profile.artifact_id}: {profile.prompt_summary}\n"
        "Protected calculus notation is binding: do not translate dr/dx/dy/dt as English abbreviations, doctor, bác sĩ, or tiến sĩ.\n"
        "Produce subtitle/display translation only. The pipeline derives spoken_text deterministically from display_text and protected notation.\n"
        if profile is not None and profile.profile_id != "generic"
        else ""
    )
    return (
        f"You translate target_segments from a {domain} educational video transcript into natural Vietnamese dubbing text.\n"
        f"{profile_text}"
        "Return raw JSON only.\n"
        "Translate only target_segments. Use context_before and context_after only to resolve meaning, pronouns, sentence continuations, terminology, and tone.\n"
        "Do not output translations for context-only segments.\n"
        "Preserve every required target segment_id exactly and return exactly one output object per required_target_segment_id.\n"
        "If a target segment begins or ends mid-sentence, translate it as a natural Vietnamese continuation that fits the neighboring context without duplicating the context.\n"
        "Do not translate word by word or preserve awkward English syntax. First understand the speaker's intent, then rephrase the idea as a natural Vietnamese explanation.\n"
        "Prefer smooth, idiomatic Vietnamese that a teacher would actually say aloud; combine, reorder, or simplify clauses when that makes the meaning clearer.\n"
        "Use locked glossary terms when their originals appear or clearly apply; preserve names, formulas, symbols, variables, units, and numbers accurately.\n"
        "Optimize for spoken Vietnamese: clear, concise, educational, and easy for TTS to read. Avoid overly literal English word order.\n"
        "Keep vi_text short enough for the segment duration when possible, but never drop technical meaning.\n"
        "Fill used_terms with glossary original terms actually used; use translation_warnings for uncertainty, missing context, or length risk."
        f"{retry_instruction}"
    )


def _translation_instruction() -> str:
    return (
        "Return exactly one JSON object with a segments array. "
        "Return exactly one object for every required target ID and no objects for context-only IDs. "
        "For each object include segment_id, vi_text, used_terms, length_ratio, and translation_warnings. "
        "You may include display_text as an alias of vi_text; do not return spoken_text."
    )


def _translation_blocks(ctx: StageContext, segments: list[dict[str, object]]) -> list[TranslationContextBlock]:
    config = ctx.config.translation
    return build_translation_context_blocks(
        segments,
        min_context_words=config.min_context_words,
        max_context_words=config.max_context_words,
        context_overlap_words=config.context_overlap_words,
        target_segment_count=config.target_segment_count,
    )


def _glossary_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term_id": {"type": "string"},
                        "original": {"type": "string"},
                        "vietnamese": {"type": "string"},
                        "category": {"type": "string"},
                        "confidence": {"type": "number"},
                        "source_segments": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "term_id",
                        "original",
                        "vietnamese",
                        "category",
                        "confidence",
                        "source_segments",
                        "notes",
                    ],
                },
            }
        },
        "required": ["terms"],
    }


def _normalize_glossary_term(
    term: object,
    *,
    block_index: int,
    term_index: int,
    block_segment_ids: list[str],
    glossary_review: bool,
) -> dict[str, object] | None:
    fallback_id = f"term_{block_index:02d}_{term_index:04d}"
    if isinstance(term, str):
        original = term.strip()
        if not original:
            return None
        return {
            "term_id": fallback_id,
            "original": original,
            "vietnamese": "",
            "category": "term",
            "confidence": 0.5,
            "locked": not glossary_review,
            "source_segments": block_segment_ids,
            "notes": "LLM returned this glossary item as a string instead of an object.",
        }
    if not isinstance(term, dict):
        logger.warning(
            "stage glossary term skipped block=%s index=%s reason=invalid_type type=%s",
            block_index,
            term_index,
            type(term).__name__,
        )
        return None
    original = str(term.get("original", "")).strip()
    if not original:
        logger.warning("stage glossary term skipped block=%s index=%s reason=empty_original", block_index, term_index)
        return None
    source_segments = term.get("source_segments", block_segment_ids)
    if not isinstance(source_segments, list):
        source_segments = block_segment_ids
    return {
        "term_id": term.get("term_id", fallback_id),
        "original": original,
        "vietnamese": term.get("vietnamese", ""),
        "category": term.get("category", "term"),
        "confidence": term.get("confidence", 1.0),
        "locked": not glossary_review,
        "source_segments": source_segments,
        "notes": term.get("notes", ""),
    }


def _fallback_glossary_terms_from_text(
    content: str,
    *,
    block_index: int,
    block_segment_ids: list[str],
    glossary_review: bool,
) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for line in content.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        is_bullet = candidate.startswith(("*", "-"))
        candidate = candidate.lstrip("-* ").strip()
        if not is_bullet and not candidate.startswith("**"):
            continue
        if candidate.startswith("**") and "**" in candidate[2:]:
            candidate = candidate[2:]
            original, _, notes = candidate.partition("**")
            original = original.strip(" :.-")
            notes = notes.lstrip(" :.-")
        elif ":" in candidate:
            original, _, notes = candidate.partition(":")
            original = original.strip(" *`.-")
            notes = notes.strip()
        else:
            continue
        if not original or len(original) > 120:
            continue
        terms.append(
            {
                "term_id": f"term_{block_index:02d}_{len(terms) + 1:04d}",
                "original": original,
                "vietnamese": "",
                "category": "term",
                "confidence": 0.4,
                "locked": not glossary_review,
                "source_segments": block_segment_ids,
                "notes": notes[:500],
            }
        )
    logger.warning("stage glossary recovered terms from non-json llm output block=%s terms=%s", block_index, len(terms))
    return terms


def _translation_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_id": {"type": "string"},
                        "vi_text": {"type": "string"},
                        "display_text": {"type": "string"},
                        "used_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "length_ratio": {"type": "number"},
                        "translation_warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "segment_id",
                        "vi_text",
                        "used_terms",
                        "length_ratio",
                        "translation_warnings",
                    ],
                },
            }
        },
        "required": ["segments"],
    }


def _source_normalization_suggestion_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_id": {"type": "string"},
                        "candidate_id": {"type": "string"},
                        "original": {"type": "string"},
                        "suggested_normalized": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "segment_id",
                        "candidate_id",
                        "original",
                        "suggested_normalized",
                        "confidence",
                        "reason",
                    ],
                },
            }
        },
        "required": ["suggestions"],
    }


def _request_structured_json(
    provider: object,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, object],
    *,
    response_name: str,
    response_description: str,
) -> Awaitable[dict[str, object]]:
    if hasattr(provider, "complete_structured_json"):
        return provider.complete_structured_json(
            system_prompt,
            user_prompt,
            schema=schema,
            response_name=response_name,
            response_description=response_description,
        )
    if hasattr(provider, "complete_json"):
        return provider.complete_json(system_prompt, user_prompt, schema=schema)
    raise AttributeError("LLM provider does not support structured JSON completion")


def _effective_domain(ctx: StageContext) -> str:
    return (ctx.config.project.domain or "general").strip() or "general"


def _effective_domain_profile(ctx: StageContext) -> DomainProfile:
    return load_domain_profile(_effective_domain(ctx), explicit_profile=ctx.config.project.domain_profile)


def _protected_spans_for_segments(segments: list[dict[str, object]], profile: DomainProfile) -> dict[str, list[ProtectedSpan]]:
    return {
        str(segment["segment_id"]): detect_protected_spans(str(segment.get("source_text", "")), profile)
        for segment in segments
    }


def _segment_with_protected_spans(segment: dict[str, object], spans: list[ProtectedSpan]) -> dict[str, object]:
    if not spans:
        return segment
    return {**segment, "protected_spans": protected_spans_prompt(spans)}


def _merge_glossary_terms(existing: list[dict[str, object]], new_terms: list[dict[str, object]]) -> list[dict[str, object]]:
    return normalize_glossary_terms([*existing, *new_terms])


def _source_normalization_suggestions(ctx: StageContext, segments: list[dict[str, object]], profile: DomainProfile) -> list[dict[str, object]]:
    if ctx.provider_mode != "openai_compatible" or not ctx.config.source_normalization.llm_adjudication:
        return []
    candidates = find_source_normalization_candidates(segments, profile)
    if not candidates:
        return []
    concurrency = ctx.require_concurrency()
    result = asyncio.run(
        concurrency.run_llm(
            lambda: _request_structured_json(
                ctx.require_provider_bundle().llm,
                (
                    "You review ASR source-normalization candidates for technical educational transcripts.\n"
                    "You only suggest possible source normalization fixes. Do not translate. Do not rewrite unrelated text.\n"
                    "The deterministic domain profile and user review remain authoritative; your output will be reviewed before TTS."
                ),
                json.dumps(
                    {
                        "instruction": (
                            "Return JSON suggestions for candidates that may be ASR mistakes. "
                            "Use suggested_normalized only for the exact technical source form, or return an empty suggestions array."
                        ),
                        "domain": _effective_domain(ctx),
                        "domain_profile": profile.artifact_id,
                        "domain_profile_summary": profile.prompt_summary,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                ),
                _source_normalization_suggestion_schema(),
                response_name="source_normalization_suggestions",
                response_description="Source normalization suggestions that require deterministic validation and human review.",
            )
        )
    )
    raw_suggestions = result.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        return []
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    segment_ids = {str(candidate["segment_id"]) for candidate in candidates}
    suggestions: list[dict[str, object]] = []
    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        if str(item.get("candidate_id", "")) not in candidate_ids:
            continue
        if str(item.get("segment_id", "")) not in segment_ids:
            continue
        suggestions.append(dict(item))
    return suggestions


def run_job_init(ctx: StageContext, copied_input: Path) -> None:
    logger.info("stage job_init started input=%s", copied_input)
    ctx.store.mark_stage(StageName.JOB_INIT, StageStatus.RUNNING)
    ctx.store.save()
    metadata_path = ctx.paths.input_dir / "input_metadata.v1.json"
    write_json_atomic(metadata_path, ctx.ffmpeg.probe(copied_input))
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
        stage=StageName.JOB_INIT,
        name="input_metadata",
        path=metadata_path,
    )
    logger.info("stage job_init completed artifact=%s", ctx.paths.to_relative(metadata_path))


def run_audio_extract(ctx: StageContext, copied_input: Path) -> int:
    logger.info("stage audio_extract started")
    ctx.store.mark_stage(StageName.AUDIO_EXTRACT, StageStatus.RUNNING)
    ctx.store.save()
    original_wav = ctx.paths.audio_dir / "original.wav"
    ctx.ffmpeg.extract_audio(copied_input, original_wav)
    shutil.copy2(original_wav, ctx.paths.audio_dir / "vocals.wav")
    duration_ms = ctx.ffmpeg.duration_ms(copied_input)
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.AUDIO_EXTRACT,
        name="audio_analysis",
        filename="audio_analysis.v1.json",
        payload={
            "schema_version": "1.0",
            "audio_duration_ms": duration_ms,
            "sample_rate": 44100,
            "channels": 1,
            "source_separation_used": False,
            "source_separation_reason": "mock_vertical_slice",
        },
    )
    logger.info("stage audio_extract completed duration_ms=%s artifact=%s", duration_ms, ctx.paths.to_relative(original_wav))
    return duration_ms


def run_vad(ctx: StageContext) -> None:
    logger.info("stage vad started mode=%s frame_ms=%s threshold_ratio=%s min_speech_duration_ms=%s preferred_max_chunk_ms=%s hard_max_chunk_ms=%s silence_merge_threshold_ms=%s context_padding_ms=%s", ctx.config.vad.mode, ctx.config.vad.frame_ms, ctx.config.vad.threshold_ratio, ctx.config.vad.min_speech_duration_ms, ctx.config.vad.preferred_max_chunk_ms, ctx.config.vad.hard_max_chunk_ms, ctx.config.vad.silence_merge_threshold_ms, ctx.config.vad.context_padding_ms)
    ctx.store.mark_stage(StageName.VAD, StageStatus.RUNNING)
    ctx.store.save()
    vad = ctx.config.vad
    segments = detect_segments(
        ctx.paths.audio_dir / "vocals.wav",
        AudioVadConfig(
            mode=vad.mode,
            frame_ms=vad.frame_ms,
            threshold_ratio=vad.threshold_ratio,
            min_duration_ms=vad.min_duration_ms,
            max_duration_ms=vad.max_duration_ms,
            min_speech_duration_ms=vad.min_speech_duration_ms,
            target_min_chunk_ms=vad.target_min_chunk_ms,
            preferred_max_chunk_ms=vad.preferred_max_chunk_ms,
            hard_max_chunk_ms=vad.hard_max_chunk_ms,
            silence_merge_threshold_ms=vad.silence_merge_threshold_ms,
            context_padding_ms=vad.context_padding_ms,
            soft_split_allowed=vad.soft_split_allowed,
            silero_model_path=vad.silero_model_path,
            silero_model_url=vad.silero_model_url,
            silero_auto_download=vad.silero_auto_download,
            silero_threshold=vad.silero_threshold,
            min_silence_duration_ms=vad.min_silence_duration_ms,
            speech_padding_ms=vad.speech_padding_ms,
            max_vad_chunk_ms=vad.max_vad_chunk_ms,
            merge_gap_ms=vad.merge_gap_ms,
        ),
    )
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.VAD,
        name="segments",
        filename="segments.v1.json",
        payload={
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "source_audio": "audio/vocals.wav",
            "segments": [segment.to_dict() for segment in segments],
        },
    )

    logger.info("stage vad completed segments=%s artifact=artifacts/segments.v1.json", len(segments))


def run_asr(ctx: StageContext) -> None:
    if ctx.provider_mode == "openai_compatible" and not ctx.config.asr_service.require_word_timestamps:
        raise ValueError(
            "production ASR requires word-level timestamps; set asr_service.require_word_timestamps=true"
        )
    segments = ctx.artifact_json("segments.v1.json")["segments"]
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    checkpoint_path = ctx.paths.artifact_path("asr_segments.v1.json")
    segment_ids = [str(segment["segment_id"]) for segment in segments]
    segment_store = SegmentCheckpointStore.load_or_create(
        checkpoint_path, stage=StageName.ASR.value, segment_ids=segment_ids
    )
    segment_store.invalidate_missing_artifacts(ctx.paths.root)
    segment_store.save()
    ctx.store.mark_stage(
        StageName.ASR,
        StageStatus.RUNNING,
        done=segment_store.done_count,
        total=len(segments),
    )
    ctx.store.save()
    source_audio = ctx.paths.audio_dir / "vocals.wav"
    provider_bundle = ctx.require_provider_bundle() if ctx.provider_mode == "openai_compatible" else None

    if ctx.provider_mode == "openai_compatible":
        for segment in segments:
            segment_id = str(segment["segment_id"])
            raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
            if segment_store.segments[segment_id].status != StageStatus.COMPLETED or not raw_path.exists():
                continue
            try:
                normalize_asr_timestamps(
                    read_json(raw_path),
                    chunk_start_ms=int(segment["start_ms"]),
                    chunk_end_ms=int(segment["end_ms"]),
                    require_timestamps=ctx.config.asr_service.require_timestamps,
                    require_word_timestamps=ctx.config.asr_service.require_word_timestamps,
                    allow_chunk_text_fallback=ctx.config.asr_service.allow_chunk_text_fallback,
                )
            except (OSError, ValueError, TypeError) as exc:
                segment_store.mark(segment_id, StageStatus.FAILED, error=str(exc))
        segment_store.save()

    pending = [
        (index, segment)
        for index, segment in enumerate(segments, start=1)
        if segment_store.segments[str(segment["segment_id"])].status != StageStatus.COMPLETED
        or not (ctx.paths.raw_dir / "asr" / f"{segment['segment_id']}.json").exists()
    ]

    if ctx.provider_mode == "openai_compatible":
        assert provider_bundle is not None
        concurrency = ctx.require_concurrency()

        def transcribe(item: tuple[int, dict[str, object]]) -> Path:
            _, segment = item
            segment_id = str(segment["segment_id"])
            raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
            clip_path = ctx.paths.audio_dir / "asr" / f"{segment_id}.wav"
            start_ms = int(segment["start_ms"])
            end_ms = int(segment["end_ms"])
            ctx.ffmpeg.extract_audio_segment(
                source_audio,
                clip_path,
                start_ms=start_ms,
                duration_ms=end_ms - start_ms,
            )
            result = asyncio.run(
                concurrency.run_asr(lambda: provider_bundle.asr.transcribe(clip_path, language="en"))
            )
            raw_payload = dict(result.raw)
            if result.confidence is not None:
                raw_payload.setdefault("confidence", result.confidence)
            write_json_atomic(raw_path, raw_payload)
            normalize_asr_timestamps(
                raw_payload,
                chunk_start_ms=start_ms,
                chunk_end_ms=end_ms,
                require_timestamps=ctx.config.asr_service.require_timestamps,
                require_word_timestamps=ctx.config.asr_service.require_word_timestamps,
                allow_chunk_text_fallback=ctx.config.asr_service.allow_chunk_text_fallback,
            )
            return raw_path

        def save_completion(completion: TaskCompletion[tuple[int, dict[str, object]], Path]) -> None:
            index, segment = completion.item
            segment_id = str(segment["segment_id"])
            if completion.error is not None:
                segment_store.mark(segment_id, StageStatus.FAILED, error=str(completion.error))
            else:
                assert completion.result is not None
                segment_store.mark(
                    segment_id,
                    StageStatus.COMPLETED,
                    artifact=ctx.paths.to_relative(completion.result),
                )
                logger.info(
                    "stage asr segment completed segment=%s index=%s/%s progress=%.1f%% eta_ms=0",
                    segment_id,
                    index,
                    len(segments),
                    (segment_store.done_count / len(segments)) * 100 if segments else 100.0,
                )
            segment_store.save()
            ctx.store.mark_stage(
                StageName.ASR,
                StageStatus.RUNNING,
                done=segment_store.done_count,
                total=len(segments),
            )
            ctx.store.save()

        logger.info("stage asr provider concurrency=%s pending=%s", concurrency.asr_limit, len(pending))
        run_bounded(pending, transcribe, max_workers=concurrency.asr_limit, on_completion=save_completion)
    else:
        for index, segment in pending:
            segment_id = str(segment["segment_id"])
            raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
            text = f"Mock transcript segment {index} for vertical slice."
            write_json_atomic(raw_path, {"text": text, "confidence": 1.0})
            segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            segment_store.save()
            ctx.store.mark_stage(
                StageName.ASR,
                StageStatus.RUNNING,
                done=segment_store.done_count,
                total=len(segments),
            )
            ctx.store.save()

    asr_chunks: list[dict[str, object]] = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
        raw = read_json(raw_path)
        confidence = float(raw.get("confidence", 1.0))
        if ctx.provider_mode == "openai_compatible":
            normalized = normalize_asr_timestamps(
                raw,
                chunk_start_ms=int(segment["start_ms"]),
                chunk_end_ms=int(segment["end_ms"]),
                require_timestamps=ctx.config.asr_service.require_timestamps,
                require_word_timestamps=ctx.config.asr_service.require_word_timestamps,
                allow_chunk_text_fallback=ctx.config.asr_service.allow_chunk_text_fallback,
            )
        else:
            text = str(raw.get("text", ""))
            normalized = NormalizedASRTimestamps(
                source="mock",
                quality="mock",
                units=[TimestampUnit(text=text, start_ms=int(segment["start_ms"]), end_ms=int(segment["end_ms"]))],
                risk_flags=[],
            )
        asr_chunks.append(
            {
                "chunk_id": segment_id,
                "raw_response_path": ctx.paths.to_relative(raw_path),
                "timestamps": normalized,
                "confidence": confidence,
                "vad_split_reason": str(segment.get("split_reason", "")),
                "vad_risk_flags": list(segment.get("risk_flags", [])),
                "silence_before_ms": int(segment.get("silence_before_ms", 0)),
                "silence_after_ms": int(segment.get("silence_after_ms", 0)),
            }
        )

    if ctx.config.asr_chunking.enabled:
        audio_duration_ms = max((int(segment["end_ms"]) for segment in segments), default=0)
        word_chunk_payloads = build_asr_word_chunk_payloads(
            asr_chunks,
            audio_duration_ms=audio_duration_ms,
            config=ctx.config.asr_chunking,
        )
        publisher.publish_json(
            stage=StageName.ASR,
            name="asr_word_chunks",
            filename="asr_word_chunks.v1.json",
            payload={
                "schema_version": "1.0",
                "source": "asr_segments.v1",
                "chunks": word_chunk_payloads["chunks"],
            },
            done=len(segments),
            total=len(segments),
        )
        asr_chunks = word_chunk_payloads["asr_chunks"]

    profile = _effective_domain_profile(ctx)
    transcript_segments = normalize_transcript_segments(
        build_transcript_segments(asr_chunks, ctx.config.transcript_segmentation),
        profile,
    )
    transcript_segments = attach_source_normalization_suggestions(
        transcript_segments,
        _source_normalization_suggestions(ctx, transcript_segments, profile),
    )
    publisher.publish_file(
        stage=StageName.ASR,
        name="asr_segments",
        path=segment_store.path,
        status=StageStatus.RUNNING,
    )
    publisher.publish_json(
        stage=StageName.ASR,
        name="transcript",
        filename="transcript.v1.json",
        payload={
            "schema_version": "1.0",
            "provider": {
                "type": ctx.provider_mode,
                "model": "provider-asr" if ctx.provider_mode == "openai_compatible" else "mock-asr",
            },
            "segments": transcript_segments,
        },
        done=len(segments),
        total=len(segments),
    )
    publisher.publish_json(
        stage=StageName.ASR,
        name="word_timeline",
        filename="word_timeline.v1.json",
        payload={
            "schema_version": "1.0",
            "source": "transcript.v1",
            "words": build_word_timeline(transcript_segments),
        },
        done=len(segments),
        total=len(segments),
    )
    publisher.publish_json(
        stage=StageName.ASR,
        name="source_normalization",
        filename="source_normalization.v1.json",
        payload={
            "schema_version": "1.0",
            "domain_profile": profile.artifact_id,
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "source_text_raw": segment.get("source_text_raw", segment.get("source_text", "")),
                    "source_text_normalized": segment.get("source_text_normalized", segment.get("source_text", "")),
                    "normalization_edits": segment.get("normalization_edits", []),
                    "normalization_suggestions": segment.get("normalization_suggestions", []),
                    "normalization_confidence": segment.get("normalization_confidence", 1.0),
                    "protected_spans": segment.get("protected_spans", []),
                    "risk_flags": segment.get("risk_flags", []),
                }
                for segment in transcript_segments
            ],
        },
        done=len(segments),
        total=len(segments),
    )


def run_glossary(ctx: StageContext, *, glossary_review: bool) -> bool:
    domain = _effective_domain(ctx)
    profile = _effective_domain_profile(ctx)
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    transcript = ctx.artifact_json("transcript.v1.json")
    blocks = _translation_blocks(ctx, transcript["segments"])
    block_ids = [f"block_{index:06d}" for index in range(1, max(1, len(blocks)) + 1)]
    checkpoint = SegmentCheckpointStore.load_or_create(
        ctx.paths.artifact_path("glossary_blocks.v1.json"),
        stage=StageName.GLOSSARY.value,
        segment_ids=block_ids,
    )
    checkpoint.invalidate_missing_artifacts(ctx.paths.root)
    checkpoint.save()
    ctx.store.mark_stage(
        StageName.GLOSSARY,
        StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    ctx.store.save()
    source_segments = [str(segment["segment_id"]) for segment in transcript["segments"]]
    transcript_span_map = _protected_spans_for_segments(transcript["segments"], profile)
    protected_glossary_terms = glossary_terms_from_spans(
        [span for spans in transcript_span_map.values() for span in spans],
        source_segments=source_segments,
        locked=True,
    )
    terms: list[dict[str, object]] = []

    if ctx.provider_mode == "openai_compatible":
        concurrency = ctx.require_concurrency()
        block_terms_by_id: dict[str, list[dict[str, object]]] = {}
        pending_blocks: list[tuple[int, TranslationContextBlock]] = []
        for block_index, block in enumerate(blocks, start=1):
            block_id = block_ids[block_index - 1]
            raw_path = ctx.paths.raw_dir / "glossary" / f"{block_id}.json"
            if checkpoint.segments[block_id].status == StageStatus.COMPLETED and raw_path.exists():
                block_terms_by_id[block_id] = list(read_json(raw_path).get("terms", []))
            else:
                pending_blocks.append((block_index, block))

        def extract_terms(item: tuple[int, TranslationContextBlock]) -> tuple[Path, list[dict[str, object]]]:
            block_index, block = item
            block_id = block_ids[block_index - 1]
            raw_path = ctx.paths.raw_dir / "glossary" / f"{block_id}.json"
            block_segments = block.all_segments
            block_segment_ids = [str(segment["segment_id"]) for segment in block_segments]
            try:
                result = asyncio.run(
                    concurrency.run_llm(
                        lambda: _request_structured_json(
                            ctx.require_provider_bundle().llm,
                            _glossary_system_prompt(domain, profile),
                            json.dumps(
                                {
                                    "instruction": _glossary_instruction(),
                                    "domain": domain,
                                    "domain_profile": profile.artifact_id,
                                    "domain_profile_summary": profile.prompt_summary,
                                    "protected_spans": {
                                        segment_id: protected_spans_prompt(transcript_span_map.get(segment_id, []))
                                        for segment_id in block_segment_ids
                                    },
                                    "segments": [
                                        _segment_with_protected_spans(
                                            dict(segment),
                                            transcript_span_map.get(str(segment["segment_id"]), []),
                                        )
                                        for segment in block_segments
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                            _glossary_response_schema(),
                            response_name="glossary_output",
                            response_description="Glossary terms extracted from the transcript.",
                        )
                    )
                )
                raw_terms = result.get("terms", [])
            except LLMStructuredOutputError as exc:
                raw_terms = _fallback_glossary_terms_from_text(
                    exc.content,
                    block_index=block_index,
                    block_segment_ids=block_segment_ids or source_segments,
                    glossary_review=glossary_review,
                )
            if not isinstance(raw_terms, list):
                raw_terms = []
            block_terms: list[dict[str, object]] = []
            for term_index, term in enumerate(raw_terms, start=1):
                normalized = _normalize_glossary_term(
                    term,
                    block_index=block_index,
                    term_index=term_index,
                    block_segment_ids=block_segment_ids or source_segments,
                    glossary_review=glossary_review,
                )
                if normalized is not None:
                    block_terms.append(normalized)
            write_json_atomic(raw_path, {"schema_version": "1.0", "terms": block_terms})
            return raw_path, block_terms

        def save_completion(
            completion: TaskCompletion[tuple[int, TranslationContextBlock], tuple[Path, list[dict[str, object]]]]
        ) -> None:
            block_index, _ = completion.item
            block_id = block_ids[block_index - 1]
            if completion.error is not None:
                checkpoint.mark(block_id, StageStatus.FAILED, error=str(completion.error))
            else:
                assert completion.result is not None
                raw_path, block_terms = completion.result
                block_terms_by_id[block_id] = block_terms
                checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()
            ctx.store.mark_stage(
                StageName.GLOSSARY,
                StageStatus.RUNNING,
                done=checkpoint.done_count,
                total=checkpoint.total_count,
            )
            ctx.store.save()

        logger.info("stage glossary provider concurrency=%s pending=%s", concurrency.llm_limit, len(pending_blocks))
        run_bounded(
            pending_blocks,
            extract_terms,
            max_workers=concurrency.llm_limit,
            on_completion=save_completion,
        )
        for block_id in block_ids:
            terms = _merge_glossary_terms(terms, block_terms_by_id.get(block_id, []))
        terms = _merge_glossary_terms(protected_glossary_terms, terms)
    else:
        block_id = block_ids[0]
        raw_path = ctx.paths.raw_dir / "glossary" / f"{block_id}.json"
        terms = [{
            "term_id": "term_0001",
            "original": "vertical slice",
            "vietnamese": "lát cắt dọc",
            "category": "phrase",
            "locked": not glossary_review,
            "source_segments": source_segments,
            "notes": "Mock glossary entry.",
        }]
        terms = _merge_glossary_terms(protected_glossary_terms, terms)
        if checkpoint.segments[block_id].status != StageStatus.COMPLETED or not raw_path.exists():
            write_json_atomic(raw_path, {"schema_version": "1.0", "terms": terms})
            checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()

    publisher.publish_file(
        stage=StageName.GLOSSARY,
        name="glossary_blocks",
        path=checkpoint.path,
        status=StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    filename = "glossary.draft.json" if glossary_review else "glossary.locked.json"
    glossary_path = ctx.paths.artifact_path(filename)
    write_json_atomic(
        glossary_path,
        {
            "schema_version": "1.0",
            "domain": domain,
            "domain_profile": profile.artifact_id,
            "status": "draft" if glossary_review else "locked",
            "terms": terms,
        },
    )
    if glossary_review:
        publisher.publish_file(
            stage=StageName.GLOSSARY,
            name="glossary_draft",
            path=glossary_path,
            status=StageStatus.WAITING_REVIEW,
        )
        return True
    publisher.publish_file(stage=StageName.GLOSSARY, name="glossary", path=glossary_path)
    return False


def run_translation(ctx: StageContext, *, total_duration_ms: int | None = None) -> bool:
    domain = _effective_domain(ctx)
    profile = _effective_domain_profile(ctx)
    transcript = ctx.artifact_json("transcript.v1.json")
    glossary = ctx.artifact_json("glossary.locked.json")
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    cue_config = ctx.config.dubbing_cues
    source_sentences = build_source_sentences(transcript["segments"]) or list(transcript["segments"])
    cues = build_dubbing_cues(
        source_sentences,
        target_duration_ms=cue_config.target_duration_ms,
        min_duration_ms=cue_config.min_duration_ms,
        max_duration_ms=cue_config.max_duration_ms,
    )
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="source_sentences",
        filename="source_sentences.v1.json",
        payload={"schema_version": "1.0", "sentences": source_sentences},
        status=StageStatus.RUNNING,
        done=len(source_sentences),
        total=len(source_sentences),
    )
    transcript_by_id = {str(segment["segment_id"]): segment for segment in transcript["segments"]}
    cue_segments = []
    cue_protected_spans: dict[str, list[ProtectedSpan]] = {}
    for cue in cues:
        cue_id = str(cue["cue_id"])
        parents = [transcript_by_id[parent_id] for parent_id in cue["parent_segment_ids"] if parent_id in transcript_by_id]
        cue["source_text_raw"] = str(cue.get("source_text_raw", "")).strip() or " ".join(
            str(parent.get("source_text_raw", parent.get("source_text", ""))).strip()
            for parent in parents
        ).strip() or str(cue.get("source_text", ""))
        normalization_edits = [
            edit
            for parent in parents
            for edit in parent.get("normalization_edits", [])
            if _normalization_item_applies_to_cue(edit, cue)
        ]
        normalization_suggestions = [
            suggestion
            for parent in parents
            for suggestion in parent.get("normalization_suggestions", [])
            if _normalization_item_applies_to_cue(suggestion, cue)
        ]
        normalization_confidences = [float(parent.get("normalization_confidence", 1.0)) for parent in parents] or [1.0]
        cue["normalization_edits"] = normalization_edits
        cue["normalization_suggestions"] = normalization_suggestions
        cue["normalization_confidence"] = min(normalization_confidences)
        cue["risk_flags"] = list(dict.fromkeys([
            *list(cue.get("risk_flags", [])),
            *(flag for parent in parents for flag in parent.get("risk_flags", [])),
        ]))
        spans = detect_protected_spans(str(cue.get("source_text", "")), profile)
        cue_protected_spans[cue_id] = spans
        cue["protected_spans"] = [span.to_dict() for span in spans]
        cue_segments.append(
            {
                "segment_id": cue_id,
                "source_text": cue["source_text"],
                "start_ms": cue["start_ms"],
                "end_ms": cue["end_ms"],
                "duration_ms": cue["duration_ms"],
                "parent_segment_ids": cue["parent_segment_ids"],
                "source_text_raw": cue["source_text_raw"],
                "normalization_edits": cue["normalization_edits"],
                "normalization_suggestions": cue["normalization_suggestions"],
                "normalization_confidence": cue["normalization_confidence"],
                "risk_flags": cue["risk_flags"],
                "protected_spans": [span.to_dict() for span in spans],
            }
        )
    review_locked = _load_review_locked(ctx)
    locked_translation_overrides = _locked_review_overrides_by_cue_id(review_locked, cues, domain_profile=profile.artifact_id)
    blocks = _translation_blocks(ctx, cue_segments)
    block_ids = [f"block_{index:06d}" for index in range(1, len(blocks) + 1)]
    checkpoint = SegmentCheckpointStore.load_or_create(
        ctx.paths.artifact_path("translation_blocks.v1.json"),
        stage=StageName.TRANSLATION.value,
        segment_ids=block_ids,
    )
    checkpoint.invalidate_missing_artifacts(ctx.paths.root)
    checkpoint.save()
    ctx.store.mark_stage(
        StageName.TRANSLATION,
        StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    ctx.store.save()
    provider_translations: dict[str, dict] = {}

    block_results: dict[str, dict[str, object]] = {}
    pending_blocks: list[tuple[int, TranslationContextBlock]] = []
    for block_index, block in enumerate(blocks, start=1):
        block_id = block_ids[block_index - 1]
        raw_path = ctx.paths.raw_dir / "translation" / f"{block_id}.json"
        if checkpoint.segments[block_id].status == StageStatus.COMPLETED and raw_path.exists():
            block_results[block_id] = read_json(raw_path)
        else:
            pending_blocks.append((block_index, block))

    if ctx.provider_mode == "openai_compatible":
        concurrency = ctx.require_concurrency()

        def translate_block(item: tuple[int, TranslationContextBlock]) -> tuple[Path, dict[str, object]]:
            block_index, block = item
            block_id = block_ids[block_index - 1]
            raw_path = ctx.paths.raw_dir / "translation" / f"{block_id}.json"
            target_ids = [str(segment["segment_id"]) for segment in block.target_segments]
            collected: dict[str, dict[str, object]] = {}
            targets_by_id = {str(item["segment_id"]): item for item in block.target_segments}
            for segment_id in target_ids:
                override = locked_translation_overrides.get(segment_id)
                if override is not None and _review_override_has_translation_text(override):
                    collected[segment_id] = _translation_item_from_review_override(segment_id, override)
            protected_retry_used: set[str] = set()
            protected_retry_messages: dict[str, list[str]] = {}
            max_attempts = max(2, ctx.config.runtime.retry_max_attempts)
            for attempt in range(1, max_attempts + 1):
                missing = [segment_id for segment_id in target_ids if segment_id not in collected]
                if not missing:
                    break
                retry_instruction = ""
                if attempt > 1:
                    retry_instruction = "\nRetry constraint: return only the still-missing segment IDs: " + ", ".join(missing)
                    protected_details = [
                        f"{segment_id}: " + "; ".join(protected_retry_messages.get(segment_id, []))
                        for segment_id in missing
                        if protected_retry_messages.get(segment_id)
                    ]
                    if protected_details:
                        retry_instruction += (
                            "\nProtected notation correction required: "
                            + " | ".join(protected_details)
                            + ". Keep every listed protected notation in the requested display/spoken form; do not translate symbols into ordinary words."
                        )
                result = asyncio.run(
                    concurrency.run_llm(
                        lambda: _request_structured_json(
                            ctx.require_provider_bundle().llm,
                            _translation_system_prompt(domain, profile, retry_instruction=retry_instruction),
                            json.dumps(
                                {
                                    "instruction": _translation_instruction(),
                                    "required_target_segment_ids": missing,
                                    "domain": domain,
                                    "domain_profile": profile.artifact_id,
                                    "domain_profile_summary": profile.prompt_summary,
                                    "target_segments": [targets_by_id[segment_id] for segment_id in missing],
                                    "context_before": block.context_before,
                                    "context_after": block.context_after,
                                    "protected_spans": {
                                        segment_id: protected_spans_prompt(cue_protected_spans.get(segment_id, []))
                                        for segment_id in missing
                                    },
                                    "glossary": glossary["terms"],
                                },
                                ensure_ascii=False,
                            ),
                            _translation_response_schema(),
                            response_name="translation_output",
                            response_description="Vietnamese translation segments.",
                        )
                    )
                )
                for item in result.get("segments", []):
                    segment_id = str(item.get("segment_id", ""))
                    if segment_id not in missing:
                        continue
                    vi_text = str(item.get("display_text") or item.get("vi_text") or "")
                    errors = protected_translation_errors(
                        str(targets_by_id[segment_id].get("source_text", "")),
                        vi_text,
                        cue_protected_spans.get(segment_id, []),
                    )
                    if errors:
                        override = locked_translation_overrides.get(segment_id)
                        if override is not None and _review_override_has_translation_text(override):
                            collected[segment_id] = _translation_item_from_review_override(segment_id, override)
                            continue
                        if segment_id in protected_retry_used:
                            raise TranslationProtectedSpanManualReviewRequired(
                                segment_id=segment_id,
                                errors=errors,
                                review_item=_translation_error_review_item(
                                    targets_by_id[segment_id],
                                    item,
                                    errors=errors,
                                    protected_spans=cue_protected_spans.get(segment_id, []),
                                ),
                            )
                        protected_retry_used.add(segment_id)
                        protected_retry_messages[segment_id] = errors
                        continue
                    collected[segment_id] = item
            missing = [segment_id for segment_id in target_ids if segment_id not in collected]
            if missing:
                raise ValueError(
                    f"Translation response missing segment_ids after {max_attempts} attempts: {', '.join(missing[:5])}"
                )
            block_result = {
                "schema_version": "1.0",
                "segments": [collected[segment_id] for segment_id in target_ids],
            }
            write_json_atomic(raw_path, block_result)
            return raw_path, block_result

        def save_completion(
            completion: TaskCompletion[tuple[int, TranslationContextBlock], tuple[Path, dict[str, object]]]
        ) -> None:
            block_index, _ = completion.item
            block_id = block_ids[block_index - 1]
            if completion.error is not None:
                checkpoint.mark(block_id, StageStatus.FAILED, error=str(completion.error))
            else:
                assert completion.result is not None
                raw_path, block_result = completion.result
                block_results[block_id] = block_result
                checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()
            ctx.store.mark_stage(
                StageName.TRANSLATION,
                StageStatus.RUNNING,
                done=checkpoint.done_count,
                total=checkpoint.total_count,
            )
            ctx.store.save()

        logger.info("stage translation provider concurrency=%s pending=%s", concurrency.llm_limit, len(pending_blocks))
        try:
            run_bounded(
                pending_blocks,
                translate_block,
                max_workers=concurrency.llm_limit,
                on_completion=save_completion,
            )
        except TranslationProtectedSpanManualReviewRequired as exc:
            _publish_translation_error_review_required(
                publisher,
                profile=profile,
                cues=cues,
                review_item=exc.review_item,
                done=checkpoint.done_count,
                total=checkpoint.total_count,
            )
            return True
    else:
        for block_index, block in pending_blocks:
            block_id = block_ids[block_index - 1]
            raw_path = ctx.paths.raw_dir / "translation" / f"{block_id}.json"
            block_result = {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": source["segment_id"],
                        "vi_text": f"Đây là lát cắt dọc thuyết minh tiếng Việt mẫu cho {source['segment_id']}.",
                        "used_terms": ["vertical slice"],
                        "length_ratio": 1.0,
                        "translation_warnings": [],
                    }
                    for source in block.target_segments
                ],
            }
            write_json_atomic(raw_path, block_result)
            checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()
            block_results[block_id] = block_result

    for block_id in block_ids:
        block_result = block_results[block_id]
        for item in block_result.get("segments", []):
            provider_translations[str(item["segment_id"])] = item

    translated_cues: list[dict[str, object]] = []
    for source in cue_segments:
        provider_segment = provider_translations.get(str(source["segment_id"]), {})
        display_text = str(provider_segment.get("display_text") or provider_segment.get("vi_text") or "")
        candidate = {
            "segment_id": source["segment_id"],
            "source_text": source["source_text"],
            "vi_text": display_text,
            "display_text": display_text,
            "spoken_text": "",
            "protected_spans": source.get("protected_spans", []),
            "used_terms": provider_segment.get("used_terms", []),
            "length_ratio": provider_segment.get("length_ratio", 1.0),
            "translation_warnings": provider_segment.get("translation_warnings", []),
        }
        compressed = compress_segment_translation(candidate, glossary["terms"], max_length_ratio=3.0)
        candidate["vi_text"] = compressed.vi_text
        candidate["display_text"] = compressed.vi_text
        candidate["spoken_text"] = normalize_spoken_text(
            str(candidate["display_text"]),
            cue_protected_spans.get(str(source["segment_id"]), []),
        )
        candidate["translation_warnings"] = list(candidate["translation_warnings"]) + compressed.warnings
        translated_cues.append(candidate)

    _trim_repeated_boundary_prefixes(translated_cues, cue_protected_spans)

    validation = validate_translations(
        cue_segments,
        translated_cues,
        glossary["terms"],
        max_length_ratio=3.0,
        protected_spans_by_segment={
            segment_id: [span.to_dict() for span in spans]
            for segment_id, spans in cue_protected_spans.items()
        },
    )
    protected_span_validation_errors: dict[str, list[str]] = {
        str(warning.get("segment_id", "")): [str(error) for error in warning.get("errors", [])]
        for warning in validation.warnings
        if warning.get("warning") == "translation_protected_span_violation"
    }
    translated_by_cue = {str(item["segment_id"]): item for item in translated_cues}
    for cue in cues:
        cue_id = str(cue["cue_id"])
        translated = translated_by_cue[cue_id]
        cue["translated_text"] = translated["display_text"]
        cue["display_text"] = translated["display_text"]
        cue["spoken_text"] = translated["spoken_text"]
        cue["protected_spans"] = translated.get("protected_spans", [])
        cue["used_terms"] = translated["used_terms"]
        cue["translation_warnings"] = translated["translation_warnings"]
        protected_errors = protected_span_validation_errors.get(cue_id, [])
        if protected_errors:
            cue["translation_protected_span_errors"] = protected_errors
            cue["risk_flags"] = list(dict.fromkeys([
                *list(cue.get("risk_flags", [])),
                "translation_protected_span_violation",
            ]))

    if ctx.provider_mode == "openai_compatible":
        _annotate_tts_timing_risks(
            cues,
            max_speedup_ratio=ctx.config.tts_service.max_speedup_ratio,
        )

    if review_locked:
        _apply_review_locked(cues, review_locked, domain_profile=profile.artifact_id)

    translated_segments: list[dict[str, object]] = []
    seen_segment_cues: set[str] = set()
    for source in transcript["segments"]:
        parent_id = str(source["segment_id"])
        children = [cue for cue in cues if parent_id in cue["parent_segment_ids"]]
        unique_children = [cue for cue in children if str(cue["cue_id"]) not in seen_segment_cues]
        segment_children = unique_children or children
        seen_segment_cues.update(str(cue["cue_id"]) for cue in segment_children)
        translated_segments.append({
            "segment_id": parent_id,
            "source_text": source["source_text"],
            "source_text_raw": source.get("source_text_raw", source["source_text"]),
            "vi_text": " ".join(str(cue["display_text"]).strip() for cue in segment_children).strip(),
            "display_text": " ".join(str(cue["display_text"]).strip() for cue in segment_children).strip(),
            "spoken_text": " ".join(str(cue["spoken_text"]).strip() for cue in segment_children).strip(),
            "protected_spans": [span for cue in segment_children for span in cue.get("protected_spans", [])],
            "used_terms": list(dict.fromkeys(term for cue in segment_children for term in cue.get("used_terms", []))),
            "length_ratio": 1.0,
            "translation_warnings": list(dict.fromkeys(warning for cue in segment_children for warning in cue.get("translation_warnings", []))),
        })
    review_items = _review_required_items(cues)
    publisher.publish_file(
        stage=StageName.TRANSLATION,
        name="translation_blocks",
        path=checkpoint.path,
        status=StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="dubbing_cues",
        filename="dubbing_cues.v2.json",
        payload={
            "schema_version": "2.0",
            "domain_profile": profile.artifact_id,
            "cues": [_dubbing_cue_v2(cue) for cue in cues],
        },
        version=2,
        schema_version="2.0",
        status=StageStatus.RUNNING,
        done=len(cues),
        total=len(cues),
    )
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="speech_timeline",
        filename="speech_timeline.v1.json",
        payload=build_speech_timeline(
            cues,
            source_segments=transcript["segments"],
            total_duration_ms=total_duration_ms,
        ),
        status=StageStatus.RUNNING,
        done=len(cues),
        total=len(cues),
    )
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="translated",
        filename="translated.v2.json",
        payload={
            "schema_version": "2.0",
            "domain_profile": profile.artifact_id,
            "segments": [_translated_segment_v2(segment) for segment in translated_segments],
            "validation_warnings": validation.warnings,
        },
        version=2,
        schema_version="2.0",
        done=len(cues),
        total=len(cues),
    )
    if review_items:
        publisher.publish_json(
            stage=StageName.TRANSLATION,
            name="review_required",
            filename="review.required.json",
            payload={
                "schema_version": "1.0",
                "domain_profile": profile.artifact_id,
                "cue_set_checksum": _cue_set_checksum(cues),
                "status": "required",
                "review_scope": "high_risk_cues",
                "cues": review_items,
            },
            status=StageStatus.WAITING_REVIEW,
            done=0,
            total=len(review_items),
        )
        return True
    return False


def _normalization_item_applies_to_cue(item: object, cue: dict[str, object]) -> bool:
    if not isinstance(item, dict):
        return False
    original = str(item.get("original", "")).strip().casefold()
    normalized = str(item.get("normalized") or item.get("suggested_normalized") or "").strip().casefold()
    if not original and not normalized:
        return True
    raw_text = str(cue.get("source_text_raw", "")).casefold()
    source_text = str(cue.get("source_text", "")).casefold()
    return bool((original and original in raw_text) or (normalized and normalized in source_text))


def _dubbing_cue_v2(cue: dict[str, object]) -> dict[str, object]:
    return {
        "cue_id": str(cue["cue_id"]),
        "start_ms": int(cue["start_ms"]),
        "end_ms": int(cue["end_ms"]),
        "duration_ms": int(cue["duration_ms"]),
        "source_text": str(cue.get("source_text", "")),
        "source_text_raw": str(cue.get("source_text_raw", cue.get("source_text", ""))),
        "display_text": str(cue.get("display_text") or cue.get("translated_text") or ""),
        "spoken_text": str(cue.get("spoken_text") or cue.get("translated_text") or ""),
        "protected_spans": list(cue.get("protected_spans", [])),
        "normalization_edits": list(cue.get("normalization_edits", [])),
        "normalization_suggestions": list(cue.get("normalization_suggestions", [])),
        "normalization_confidence": float(cue.get("normalization_confidence", 1.0)),
        "risk_flags": list(cue.get("risk_flags", [])),
        "parent_segment_ids": list(cue.get("parent_segment_ids", [])),
        "source_chunk_ids": list(cue.get("source_chunk_ids", [])),
        "used_terms": list(cue.get("used_terms", [])),
        "translation_warnings": list(cue.get("translation_warnings", [])),
    }


def _review_required_items(cues: list[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for cue in cues:
        if cue.get("review_status") == "locked":
            continue
        normalization_edits = list(cue.get("normalization_edits", []))
        normalization_suggestions = list(cue.get("normalization_suggestions", []))
        risk_flags = list(cue.get("risk_flags", []))
        high_risk_flags = [flag for flag in risk_flags if flag in _HIGH_RISK_CUE_FLAGS]
        protected_errors = [
            str(error)
            for error in cue.get("translation_protected_span_errors", [])
            if str(error).strip()
        ]
        if not normalization_edits and not normalization_suggestions and not high_risk_flags and not protected_errors:
            continue
        reason = (
            "translation_protected_span_review_required"
            if protected_errors
            else "source_normalization_review_required"
            if normalization_edits or normalization_suggestions
            else "asr_timeline_review_required"
        )
        item = {
            "cue_id": str(cue["cue_id"]),
            "reason": reason,
            "source_text_raw": str(cue.get("source_text_raw", cue.get("source_text", ""))),
            "source_text_normalized": str(cue.get("source_text", "")),
            "display_text": str(cue.get("display_text") or cue.get("translated_text") or ""),
            "spoken_text": str(cue.get("spoken_text") or cue.get("translated_text") or ""),
            "start_ms": int(cue.get("start_ms", 0)),
            "end_ms": int(cue.get("end_ms", cue.get("start_ms", 0))),
            "duration_ms": int(cue.get("duration_ms", int(cue.get("end_ms", cue.get("start_ms", 0))) - int(cue.get("start_ms", 0)))),
            "protected_spans": list(cue.get("protected_spans", [])),
            "normalization_edits": normalization_edits,
            "normalization_suggestions": normalization_suggestions,
            "risk_flags": risk_flags,
            "review_overrides": {
                "source_text_normalized": str(cue.get("source_text", "")),
                "display_text": str(cue.get("display_text") or cue.get("translated_text") or ""),
                "spoken_text": str(cue.get("spoken_text") or cue.get("translated_text") or ""),
                "protected_spans": list(cue.get("protected_spans", [])),
                "start_ms": int(cue.get("start_ms", 0)),
                "end_ms": int(cue.get("end_ms", cue.get("start_ms", 0))),
            },
        }
        if protected_errors:
            item["error"] = "; ".join(protected_errors)
        items.append(item)
    return sorted(items, key=_review_item_sort_key)


def _review_item_sort_key(item: dict[str, object]) -> tuple[int, int, str]:
    return (
        int(item.get("start_ms", 0)),
        int(item.get("end_ms", item.get("start_ms", 0))),
        str(item.get("cue_id", "")),
    )


def _translation_error_review_item(
    source: dict[str, object],
    provider_item: dict[str, object],
    *,
    errors: list[str],
    protected_spans: list[ProtectedSpan],
) -> dict[str, object]:
    cue_id = str(source["segment_id"])
    display_text = str(provider_item.get("display_text") or provider_item.get("vi_text") or "")
    spoken_text = normalize_spoken_text(display_text, protected_spans) if display_text else ""
    protected = [span.to_dict() for span in protected_spans]
    start_ms = int(source.get("start_ms", 0))
    end_ms = int(source.get("end_ms", start_ms))
    return {
        "cue_id": cue_id,
        "reason": "translation_protected_span_review_required",
        "error": "; ".join(errors),
        "source_text_raw": str(source.get("source_text_raw", source.get("source_text", ""))),
        "source_text_normalized": str(source.get("source_text", "")),
        "display_text": display_text,
        "spoken_text": spoken_text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": int(source.get("duration_ms", end_ms - start_ms)),
        "protected_spans": protected,
        "risk_flags": list(dict.fromkeys([*list(source.get("risk_flags", [])), "translation_protected_span_violation"])),
        "review_overrides": {
            "source_text_normalized": str(source.get("source_text", "")),
            "display_text": display_text,
            "spoken_text": spoken_text,
            "protected_spans": protected,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
    }


def _publish_translation_error_review_required(
    publisher: StageArtifacts,
    *,
    profile: DomainProfile,
    cues: list[dict[str, object]],
    review_item: dict[str, object],
    done: int,
    total: int,
) -> None:
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="review_required",
        filename="review.required.json",
        payload={
            "schema_version": "1.0",
            "domain_profile": profile.artifact_id,
            "cue_set_checksum": _cue_set_checksum(cues),
            "status": "required",
            "review_scope": "translation_error",
            "cues": [review_item],
        },
        status=StageStatus.WAITING_REVIEW,
        done=done,
        total=total,
    )


def _locked_review_overrides_by_cue_id(
    review_locked: dict[str, object],
    cues: list[dict[str, object]],
    *,
    domain_profile: str = "",
) -> dict[str, dict[str, object]]:
    locked_profile = str(review_locked.get("domain_profile", "")).strip()
    if locked_profile and domain_profile and locked_profile != domain_profile:
        return {}
    expected_checksum = str(review_locked.get("cue_set_checksum", "")).strip()
    if expected_checksum and expected_checksum != _cue_set_checksum(cues):
        return {}
    locked_items = review_locked.get("cues", [])
    if not isinstance(locked_items, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for item in locked_items:
        if not isinstance(item, dict):
            continue
        cue_id = str(item.get("cue_id", ""))
        overrides = item.get("review_overrides", item)
        if cue_id and isinstance(overrides, dict):
            result[cue_id] = dict(overrides)
    return result


def _review_override_has_translation_text(override: dict[str, object]) -> bool:
    return any(str(override.get(key, "")).strip() for key in ("display_text", "vi_text", "spoken_text"))


def _translation_item_from_review_override(segment_id: str, override: dict[str, object]) -> dict[str, object]:
    display_text = str(override.get("display_text") or override.get("vi_text") or "")
    warnings = override.get("translation_warnings", [])
    return {
        "segment_id": segment_id,
        "vi_text": display_text,
        "display_text": display_text,
        "spoken_text": str(override.get("spoken_text") or display_text),
        "used_terms": list(override.get("used_terms", [])) if isinstance(override.get("used_terms", []), list) else [],
        "length_ratio": float(override.get("length_ratio", 1.0) or 1.0),
        "translation_warnings": list(dict.fromkeys([
            *(warnings if isinstance(warnings, list) else []),
            "manual_review_locked",
        ])),
    }


def _annotate_tts_timing_risks(
    cues: list[dict[str, object]],
    *,
    max_speedup_ratio: float,
    estimated_token_ms: int = 300,
    guard_ms: int = 100,
) -> None:
    for index, cue in enumerate(cues):
        spoken_text = str(cue.get("spoken_text") or cue.get("translated_text") or "").strip()
        token_count = len(re.findall(r"\w+", spoken_text, flags=re.UNICODE))
        if token_count == 0:
            continue
        start_ms = int(cue.get("start_ms", 0))
        end_ms = int(cue.get("end_ms", start_ms))
        next_start_ms = (
            int(cues[index + 1].get("start_ms", end_ms))
            if index + 1 < len(cues)
            else end_ms
        )
        available_ms = max(end_ms - start_ms, next_start_ms - start_ms - guard_ms)
        estimated_duration_ms = token_count * estimated_token_ms
        required_ratio = estimated_duration_ms / max(1, available_ms)
        cue["tts_timing_estimate_ms"] = estimated_duration_ms
        cue["tts_timing_available_ms"] = available_ms
        cue["tts_timing_required_speedup_ratio"] = round(required_ratio, 4)
        if required_ratio > max_speedup_ratio:
            cue["risk_flags"] = list(dict.fromkeys([
                *list(cue.get("risk_flags", [])),
                "tts_timing_density_high",
            ]))


def _load_review_locked(ctx: StageContext) -> dict[str, object]:
    path = ctx.paths.artifact_path("review.locked.json")
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _apply_review_locked(
    cues: list[dict[str, object]],
    review_locked: dict[str, object],
    *,
    domain_profile: str = "",
) -> None:
    locked_profile = str(review_locked.get("domain_profile", "")).strip()
    if locked_profile and domain_profile and locked_profile != domain_profile:
        return
    expected_checksum = str(review_locked.get("cue_set_checksum", "")).strip()
    if expected_checksum and expected_checksum != _cue_set_checksum(cues):
        return
    locked_items = review_locked.get("cues", [])
    if not isinstance(locked_items, list):
        return
    by_id = {str(item.get("cue_id")): item for item in locked_items if isinstance(item, dict)}
    timing_updates: dict[str, tuple[int, int]] = {}
    for index, cue in enumerate(cues):
        item = by_id.get(str(cue["cue_id"]))
        if not item:
            continue
        overrides = item.get("review_overrides", item)
        if not isinstance(overrides, dict):
            continue
        if "source_text_normalized" in overrides:
            cue["source_text"] = str(overrides["source_text_normalized"])
        if "display_text" in overrides:
            cue["display_text"] = str(overrides["display_text"])
            cue["translated_text"] = str(overrides["display_text"])
        if "spoken_text" in overrides:
            cue["spoken_text"] = str(overrides["spoken_text"])
        if "protected_spans" in overrides and isinstance(overrides["protected_spans"], list):
            cue["protected_spans"] = list(overrides["protected_spans"])
        if "start_ms" in overrides or "end_ms" in overrides:
            start_ms = _review_timing_value(
                overrides.get("start_ms", cue["start_ms"]),
                cue_id=str(cue["cue_id"]),
                field="start_ms",
            )
            end_ms = _review_timing_value(
                overrides.get("end_ms", cue["end_ms"]),
                cue_id=str(cue["cue_id"]),
                field="end_ms",
            )
            _validate_review_timing_override(cues, index, start_ms, end_ms)
            timing_updates[str(cue["cue_id"])] = (start_ms, end_ms)
        cue["review_status"] = "locked"
    for cue in cues:
        update = timing_updates.get(str(cue["cue_id"]))
        if update is None:
            continue
        start_ms, end_ms = update
        cue["start_ms"] = start_ms
        cue["end_ms"] = end_ms
        cue["duration_ms"] = end_ms - start_ms


def _cue_set_checksum(cues: list[dict[str, object]]) -> str:
    payload = [
        {
            "cue_id": str(cue.get("cue_id", "")),
            "start_ms": int(cue.get("start_ms", 0)),
            "end_ms": int(cue.get("end_ms", 0)),
            "source_text": str(cue.get("source_text", "")),
            "display_text": str(cue.get("display_text") or cue.get("translated_text") or ""),
            "spoken_text": str(cue.get("spoken_text") or cue.get("translated_text") or ""),
        }
        for cue in cues
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _review_timing_value(value: object, *, cue_id: str, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{cue_id}: review timing override {field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{cue_id}: review timing override {field} must be a non-negative integer") from None
    if parsed < 0:
        raise ValueError(f"{cue_id}: review timing override {field} must be a non-negative integer")
    return parsed


def _validate_review_timing_override(cues: list[dict[str, object]], index: int, start_ms: int, end_ms: int) -> None:
    cue_id = str(cues[index]["cue_id"])
    if end_ms <= start_ms:
        raise ValueError(f"{cue_id}: review timing override must satisfy start_ms < end_ms")
    if index > 0:
        previous_end = int(cues[index - 1]["end_ms"])
        if start_ms < previous_end:
            raise ValueError(f"{cue_id}: review timing override overlaps previous cue")
    if index + 1 < len(cues):
        next_start = int(cues[index + 1]["start_ms"])
        if end_ms > next_start:
            raise ValueError(f"{cue_id}: review timing override overlaps next cue")


def _trim_repeated_boundary_prefixes(
    cues: list[dict[str, object]],
    cue_protected_spans: dict[str, list[ProtectedSpan]],
) -> None:
    previous_text = ""
    for cue in cues:
        current_text = str(cue.get("display_text") or "").strip()
        if previous_text and current_text:
            trimmed_text, trimmed = _trim_repeated_boundary_prefix(previous_text, current_text)
            if trimmed and trimmed_text != current_text:
                cue["display_text"] = trimmed_text
                cue["vi_text"] = trimmed_text
                cue["spoken_text"] = normalize_spoken_text(
                    trimmed_text,
                    cue_protected_spans.get(str(cue.get("segment_id")), []),
                )
                warnings = list(cue.get("translation_warnings", []))
                warnings.append("boundary_overlap_trimmed")
                cue["translation_warnings"] = list(dict.fromkeys(warnings))
                current_text = trimmed_text
        previous_text = current_text


def _trim_repeated_boundary_prefix(previous_text: str, current_text: str) -> tuple[str, bool]:
    previous_words = _boundary_words(previous_text)
    current_words = _boundary_words(current_text)
    if len(previous_words) < 5 or len(current_words) < 5:
        return current_text, False

    if len(current_words) <= len(previous_words):
        return current_text, False

    max_overlap = min(len(previous_words), len(current_words))
    for overlap in range(max_overlap, 4, -1):
        if previous_words[:overlap] != current_words[:overlap]:
            continue
        cutoff = _word_cutoff_after(current_text, overlap)
        if cutoff is None:
            continue
        trimmed = current_text[cutoff:].lstrip(" \t\r\n,;:!?—–-")
        if len(_boundary_words(trimmed)) < 1:
            continue
        return trimmed, True
    return current_text, False

def _boundary_words(text: str) -> list[str]:
    return [
        match.group(0).casefold()
        for match in _BOUNDARY_WORD_RE.finditer(text)
        if re.search(r"\w", match.group(0), flags=re.UNICODE)
    ]


def _word_cutoff_after(text: str, word_count: int) -> int | None:
    seen = 0
    for match in _BOUNDARY_WORD_RE.finditer(text):
        if not re.search(r"\w", match.group(0), flags=re.UNICODE):
            continue
        seen += 1
        if seen == word_count:
            return match.end()
    return None


def _translated_segment_v2(segment: dict[str, object]) -> dict[str, object]:
    return {
        "segment_id": str(segment["segment_id"]),
        "source_text": str(segment.get("source_text", "")),
        "source_text_raw": str(segment.get("source_text_raw", segment.get("source_text", ""))),
        "display_text": str(segment.get("display_text") or segment.get("vi_text") or ""),
        "spoken_text": str(segment.get("spoken_text") or segment.get("vi_text") or ""),
        "protected_spans": list(segment.get("protected_spans", [])),
        "used_terms": list(segment.get("used_terms", [])),
        "translation_warnings": list(segment.get("translation_warnings", [])),
    }


def _timeline_overflow_by_cue(speech_timeline: dict[str, object]) -> dict[str, int]:
    overflow_by_cue: dict[str, int] = {}
    gaps = speech_timeline.get("silence_intervals", [])
    if not isinstance(gaps, list):
        return overflow_by_cue
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        cue_id = gap.get("preceding_cue_id")
        if cue_id is None:
            continue
        try:
            usable = int(gap.get("usable_overflow_ms", 0))
        except (TypeError, ValueError):
            usable = 0
        overflow_by_cue[str(cue_id)] = max(overflow_by_cue.get(str(cue_id), 0), max(0, usable))
    return overflow_by_cue


def run_tts(
    ctx: StageContext,
    duration_ms: int,
    *,
    crash_stage: str | None = None,
    crash_after_segments: int | None = None,
    tts_review_handler: Callable[[TTSReviewRequest], TTSReviewDecision] | None = None,
) -> Path:
    cues = ctx.artifact_json("dubbing_cues.v2.json")["cues"]
    timeline_overflow_by_cue = _timeline_overflow_by_cue(ctx.artifact_json("speech_timeline.v1.json"))
    segments = [
        {
            **cue,
            "segment_id": cue["cue_id"],
            "source_text": cue["source_text"],
            "timeline_usable_overflow_ms": timeline_overflow_by_cue.get(str(cue["cue_id"]), 0),
        }
        for cue in cues
    ]
    _apply_tts_overrides(ctx, segments)
    segment_ids = [str(segment["segment_id"]) for segment in segments]
    checkpoint = SegmentCheckpointStore.load_or_create(
        ctx.paths.artifact_path("tts_segments.v1.json"),
        stage=StageName.TTS.value,
        segment_ids=segment_ids,
    )
    checkpoint.invalidate_missing_artifacts(ctx.paths.root)
    checkpoint.save()
    ctx.store.mark_stage(
        StageName.TTS,
        StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=len(segments),
    )
    ctx.store.save()
    if crash_stage == StageName.TTS.value and crash_after_segments == 0:
        raise RuntimeError("simulated crash at tts after 0 segments")

    manifest_path = ctx.paths.artifact_path("tts_manifest.v1.json")
    existing_rows = list(read_json(manifest_path).get("segments", [])) if manifest_path.exists() else []
    existing_by_id = {str(row["segment_id"]): row for row in existing_rows}

    rows_by_id: dict[str, dict[str, object]] = {}
    pending: list[tuple[int, dict[str, object]]] = []
    for index, segment in enumerate(segments):
        cue_id = str(segment["segment_id"])
        prior_row = existing_by_id.get(cue_id)
        reusable = (
            checkpoint.segments[cue_id].status == StageStatus.COMPLETED
            and prior_row is not None
            and ctx.paths.resolve_relative(str(prior_row["audio_path"])).exists()
        )
        if reusable:
            rows_by_id[cue_id] = prior_row
        else:
            pending.append((index, segment))

    if ctx.provider_mode == "openai_compatible":
        tts_config = ctx.config.tts_service
        concurrency = ctx.require_concurrency()
        worker_count = max(concurrency.asr_limit, concurrency.llm_limit, concurrency.tts_limit)
        completed_count = 0

        def produce(item: tuple[int, dict[str, object]]) -> dict[str, object]:
            index, segment = item
            tts_text = _tts_text_for_segment(segment)
            return asyncio.run(
                produce_provider_tts_segment(
                    paths=ctx.paths,
                    segment=segment,
                    text=tts_text if str(segment.get("source_text", "")).strip() else "",
                    provider_bundle=ctx.require_provider_bundle(),
                    ffmpeg=ctx.ffmpeg,
                    quality_retry_attempts=tts_config.quality_retry_attempts,
                    rephrase_attempts=tts_config.rephrase_attempts,
                    max_speedup_ratio=tts_config.max_speedup_ratio,
                    min_rms=tts_config.min_rms,
                    silence_rms_threshold=tts_config.silence_rms_threshold,
                    max_edge_silence_ms=tts_config.max_edge_silence_ms,
                    max_internal_silence_ms=tts_config.max_internal_silence_ms,
                    clipping_peak_threshold=tts_config.clipping_peak_threshold,
                    max_clipped_sample_ratio=tts_config.max_clipped_sample_ratio,
                    max_overflow_ms=min(
                        tts_config.max_overflow_ms,
                        int(segment.get("timeline_usable_overflow_ms", 0)),
                    ),
                    overflow_reserve_ms=0,
                    start_delay_ms=tts_config.start_delay_ms,
                    retained_edge_silence_ms=tts_config.retained_edge_silence_ms,
                    semantic_max_cer=tts_config.semantic_max_cer,
                    semantic_min_token_recall=tts_config.semantic_min_token_recall,
                    semantic_retry_attempts=tts_config.semantic_retry_attempts,
                    next_segment_start_ms=(
                        int(segments[index + 1]["start_ms"]) + tts_config.start_delay_ms
                        if index + 1 < len(segments) else duration_ms
                    ),
                    concurrency=concurrency,
                )
            )

        def save_manifest() -> None:
            ordered_rows = [rows_by_id[segment_id] for segment_id in segment_ids if segment_id in rows_by_id]
            write_json_atomic(manifest_path, {"schema_version": "1.0", "segments": ordered_rows})

        manual_failures: list[tuple[dict[str, object], dict[str, object], BaseException]] = []

        def save_completion(
            completion: TaskCompletion[tuple[int, dict[str, object]], dict[str, object]]
        ) -> None:
            nonlocal completed_count
            _, segment = completion.item
            cue_id = str(segment["segment_id"])
            if completion.error is not None:
                checkpoint.mark(cue_id, StageStatus.FAILED, error=str(completion.error))
                quality_path = ctx.paths.raw_dir / "tts" / f"{cue_id}.quality.json"
                failed_row: dict[str, object] = read_json(quality_path) if quality_path.exists() else {}
                failed_row.update({
                    "cue_id": cue_id,
                    "segment_id": cue_id,
                    "status": "failed",
                    "final_text": str(failed_row.get("final_text", _tts_text_for_segment(segment))),
                    "display_text": _tts_display_text_for_segment(segment),
                    "spoken_text": _tts_text_for_segment(segment),
                    "final_error": str(completion.error),
                    "source_text": segment["source_text"],
                    "parent_segment_ids": list(segment.get("parent_segment_ids", [])),
                })
                rows_by_id[cue_id] = failed_row
                if is_manual_tts_review_error(completion.error):
                    manual_failures.append((segment, failed_row, completion.error))
            else:
                assert completion.result is not None
                row = completion.result
                row["cue_id"] = cue_id
                row["source_text"] = segment["source_text"]
                row["display_text"] = _tts_display_text_for_segment(segment)
                row["spoken_text"] = _tts_text_for_segment(segment)
                row["parent_segment_ids"] = list(segment.get("parent_segment_ids", []))
                rows_by_id[cue_id] = row
                checkpoint.mark(cue_id, StageStatus.COMPLETED, artifact=str(row["audio_path"]))
                completed_count += 1
            checkpoint.save()
            save_manifest()
            ctx.store.mark_stage(
                StageName.TTS,
                StageStatus.RUNNING,
                done=checkpoint.done_count,
                total=len(segments),
            )
            ctx.store.save()
            if (
                crash_stage == StageName.TTS.value
                and crash_after_segments is not None
                and completed_count == crash_after_segments
            ):
                raise RuntimeError(f"simulated crash at tts after {completed_count} segments")

        while True:
            pending = [
                (index, segment)
                for index, segment in enumerate(segments)
                if checkpoint.segments[str(segment["segment_id"])].status != StageStatus.COMPLETED
            ]
            if not pending:
                break
            failed_pending = sum(
                1
                for _, segment in pending
                if checkpoint.segments[str(segment["segment_id"])].status == StageStatus.FAILED
            )
            logger.info(
                "stage tts provider concurrency tts=%s asr=%s llm=%s cue_workers=%s reused=%s pending=%s failed_pending=%s",
                concurrency.tts_limit,
                concurrency.asr_limit,
                concurrency.llm_limit,
                worker_count,
                len(rows_by_id),
                len(pending),
                failed_pending,
            )
            manual_failures.clear()
            try:
                run_bounded(pending, produce, max_workers=worker_count, on_completion=save_completion)
            except BaseException:
                if tts_review_handler is None or not manual_failures:
                    raise
                failed_segment, failed_row, failed_error = manual_failures[0]
                cue_id = str(failed_segment["segment_id"])
                decision = tts_review_handler(
                    TTSReviewRequest(
                        paths=ctx.paths,
                        cue=failed_segment,
                        all_cues=segments,
                        failed_row=failed_row,
                    )
                )
                if decision.action in {"fail", "quit"}:
                    raise RuntimeError(f"{cue_id}: manual TTS review {decision.action}") from failed_error
                if decision.action not in {"edit", "silence"}:
                    raise RuntimeError(f"{cue_id}: unsupported manual TTS review action {decision.action!r}") from failed_error
                save_tts_interactive_override(
                    ctx.paths,
                    cue_id=cue_id,
                    action=decision.action,
                    display_text=decision.display_text,
                    spoken_text=decision.spoken_text,
                )
                _apply_tts_override(
                    failed_segment,
                    {
                        "action": decision.action,
                        "display_text": decision.display_text,
                        "spoken_text": decision.spoken_text,
                    },
                )
                rows_by_id.pop(cue_id, None)
                checkpoint.mark(cue_id, StageStatus.PENDING)
                checkpoint.save()
                save_manifest()
                continue
            break
    else:
        for index, segment in pending:
            cue_id = str(segment["segment_id"])
            row = produce_mock_tts_segment(paths=ctx.paths, segment=segment)
            spoken_text = _tts_text_for_segment(segment)
            display_text = _tts_display_text_for_segment(segment)
            row["final_text"] = spoken_text
            row["display_text"] = display_text
            row["spoken_text"] = spoken_text
            row["semantic_metrics"] = {
                "expected_text": spoken_text,
                "transcript": spoken_text,
                "cer": 0.0,
                "token_recall": 1.0,
                "mock": True,
            }
            row["cue_id"] = cue_id
            row["source_text"] = segment["source_text"]
            row["parent_segment_ids"] = list(segment.get("parent_segment_ids", []))
            rows_by_id[cue_id] = row
            checkpoint.mark(cue_id, StageStatus.COMPLETED, artifact=str(row["audio_path"]))
            checkpoint.save()
            write_json_atomic(
                manifest_path,
                {
                    "schema_version": "1.0",
                    "segments": [rows_by_id[item_id] for item_id in segment_ids if item_id in rows_by_id],
                },
            )
            ctx.store.mark_stage(
                StageName.TTS,
                StageStatus.RUNNING,
                done=checkpoint.done_count,
                total=len(segments),
            )
            ctx.store.save()
            if crash_stage == StageName.TTS.value and crash_after_segments == index + 1:
                raise RuntimeError(f"simulated crash at tts after {index + 1} segments")

    tts_segments = [rows_by_id[segment_id] for segment_id in segment_ids]

    mix_audio_path = ctx.paths.tts_dir / "mix.wav"
    assemble_commentary_track(
        paths=ctx.paths,
        ffmpeg=ctx.ffmpeg,
        tts_segments=tts_segments,
        output_audio=mix_audio_path,
        max_speedup_ratio=ctx.config.tts_service.max_speedup_ratio,
    )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(
        stage=StageName.TTS,
        name="tts_segments",
        path=checkpoint.path,
        status=StageStatus.RUNNING,
    )
    publisher.publish_json(
        stage=StageName.TTS,
        name="tts_manifest",
        filename="tts_manifest.v1.json",
        payload={"schema_version": "1.0", "segments": tts_segments},
        done=len(segments),
        total=len(segments),
    )
    publisher.publish_json(
        stage=StageName.TTS,
        name="spoken_cues",
        filename="spoken_cues.v1.json",
        payload={"schema_version": "1.0", "cues": tts_segments},
        done=len(segments),
        total=len(segments),
    )
    return mix_audio_path


def _tts_text_for_segment(segment: dict[str, object]) -> str:
    if segment.get("tts_manual_action") == "silence":
        return ""
    if "spoken_text" in segment:
        return str(segment.get("spoken_text") or "")
    return str(segment.get("translated_text") or "")


def _tts_display_text_for_segment(segment: dict[str, object]) -> str:
    if segment.get("tts_manual_action") == "silence":
        return ""
    if "display_text" in segment:
        return str(segment.get("display_text") or "")
    return str(segment.get("translated_text") or "")


def _apply_tts_overrides(ctx: StageContext, segments: list[dict[str, object]]) -> None:
    overrides = load_tts_interactive_overrides(ctx.paths)
    for segment in segments:
        override = overrides.get(str(segment["segment_id"]))
        if override is not None:
            _apply_tts_override(segment, override)


def _apply_tts_override(segment: dict[str, object], override: dict[str, object]) -> None:
    action = str(override.get("action") or "edit")
    segment["tts_manual_action"] = action
    segment["display_text"] = str(override.get("display_text") or "")
    segment["spoken_text"] = str(override.get("spoken_text") or "")


def run_mixing(ctx: StageContext, copied_input: Path, tts_audio: Path, duration_ms: int) -> Path:
    logger.info("stage mixing started tts_audio=%s", tts_audio)
    ctx.store.mark_stage(StageName.MIXING, StageStatus.RUNNING)
    ctx.store.save()
    final_audio = ctx.paths.audio_dir / "final_mix.wav"
    ctx.ffmpeg.mix_commentary_audio(
        ctx.paths.audio_dir / "original.wav",
        tts_audio,
        final_audio,
        original_ducking_db=ctx.config.mixing.original_ducking_db,
        tts_boost_db=ctx.config.mixing.tts_boost_db,
        final_loudness_normalization=ctx.config.mixing.final_loudness_normalization,
    )
    output_video = ctx.paths.output_dir / f"{copied_input.stem}_vi.mp4"
    muxed_video = output_video
    if ctx.config.subtitles.enabled:
        muxed_video = ctx.paths.output_dir / f"{copied_input.stem}_vi.mux.mp4"
    ctx.ffmpeg.mux_video_audio(copied_input, final_audio, muxed_video)
    subtitle_ass_path: Path | None = None
    if ctx.config.subtitles.enabled:
        if ctx.config.subtitles.mode != "burn_in":
            raise ValueError(f"Unsupported subtitles.mode: {ctx.config.subtitles.mode}")
        subtitle_ass_path = ctx.paths.artifact_path("subtitles.ass")
        video_width, video_height = _video_dimensions(ctx.ffmpeg.probe(muxed_video))
        subtitle_ass_path.write_text(
            render_ass(
                build_spoken_subtitle_cues(ctx.artifact_json("spoken_cues.v1.json")),
                ctx.config.subtitles,
                video_width=video_width,
                video_height=video_height,
            ),
            encoding="utf-8",
        )
        ctx.ffmpeg.burn_in_subtitles(muxed_video, subtitle_ass_path, output_video)
    qa_path = ctx.paths.output_dir / f"{copied_input.stem}_vi.qa.json"
    tts_manifest_path = ctx.paths.artifact_path("tts_manifest.v1.json")
    tts_manifest = read_json(tts_manifest_path) if tts_manifest_path.exists() else {"segments": []}
    tts_segments = list(tts_manifest.get("segments", []))
    glossary_terms = list(ctx.artifact_json("glossary.locked.json").get("terms", []))
    overflow_values = [int(segment.get("overflow_ms", 0)) for segment in tts_segments]
    drift_values = [
        max(
            abs(int(segment.get("target_start_ms", 0)) + int(segment.get("quality_report", {}).get("leading_silence_ms", 0)) - int(segment.get("original_start_ms", 0))),
            max(0, int(segment.get("target_end_ms", 0)) - int(segment.get("original_end_ms", 0))),
        )
        for segment in tts_segments
    ]
    cer_values = [float(segment.get("semantic_metrics", {}).get("cer", 0.0)) for segment in tts_segments]
    token_recalls = [float(segment.get("semantic_metrics", {}).get("token_recall", 1.0)) for segment in tts_segments]
    semantic_retries = sum(
        1
        for segment in tts_segments
        for attempt in segment.get("quality_attempts", [])
        if attempt.get("semantic_ok") is False
    )
    unused_window_values = [
        max(0, int(segment.get("target_window_ms", 0)) - int(segment.get("tts_duration_ms", 0) / max(1.0, float(segment.get("stretch_ratio", 1.0)))))
        for segment in tts_segments
    ]
    write_json_atomic(
        qa_path,
        {
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "input_duration_ms": duration_ms,
            "output_duration_ms": ctx.ffmpeg.duration_ms(output_video),
            "segments_total": len(ctx.artifact_json("segments.v1.json")["segments"]),
            "cues_total": len(tts_segments),
            "low_confidence_segments": 0,
            "glossary_terms": len(glossary_terms),
            "tts_overflow_segments": sum(1 for value in overflow_values if value > 0),
            "max_overflow_ms": max(overflow_values, default=0),
            "sync_drift_p95_ms": _percentile(drift_values, 0.95),
            "rephrased_cues": sum(1 for segment in tts_segments if int(segment.get("rephrase_attempts", 0)) > 0),
            "semantic_retries": semantic_retries,
            "semantic_failures": sum(1 for segment in tts_segments if not bool(segment.get("semantic_metrics"))),
            "semantic_cer_p95": _percentile(cer_values, 0.95),
            "semantic_token_recall_min": min(token_recalls, default=1.0),
            "unused_window_total_ms": sum(unused_window_values),
            "overflow_total_ms": sum(overflow_values),
            "warnings": [],
        },
    )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(
        stage=StageName.MIXING,
        name="final_audio",
        path=final_audio,
        status=StageStatus.RUNNING,
    )
    publisher.publish_file(
        stage=StageName.MIXING,
        name="qa_report",
        path=qa_path,
        status=StageStatus.RUNNING,
    )
    if subtitle_ass_path is not None and ctx.config.subtitles.output_sidecar:
        publisher.publish_file(
            stage=StageName.MIXING,
            name="subtitle_ass",
            path=subtitle_ass_path,
            status=StageStatus.RUNNING,
        )
    publisher.publish_file(
        stage=StageName.MIXING,
        name="output_video",
        path=output_video,
        status=StageStatus.COMPLETED,
    )
    logger.info("stage mixing completed output=%s qa=%s", output_video, qa_path)
    return output_video


def _video_dimensions(metadata: dict[str, object]) -> tuple[int, int]:
    streams = metadata.get("streams", [])
    if not isinstance(streams, list):
        return 1920, 1080
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        try:
            width = int(stream.get("width", 1920))
            height = int(stream.get("height", 1080))
        except (TypeError, ValueError):
            return 1920, 1080
        if width > 0 and height > 0:
            return width, height
    return 1920, 1080


def _percentile(values: list[int] | list[float], quantile: float) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]
