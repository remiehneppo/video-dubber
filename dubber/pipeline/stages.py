from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable
from time import perf_counter
from pathlib import Path

from dubber.asr.timestamps import NormalizedASRTimestamps, TimestampUnit, normalize_asr_timestamps
from dubber.audio.vad import VadConfig as AudioVadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext
from dubber.providers.llm_openai_compatible import LLMStructuredOutputError
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_segment
from dubber.tts.track_mixer import assemble_commentary_track
from dubber.transcript.segmentation import build_transcript_segments
from dubber.translation.block_builder import build_translation_blocks
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations

logger = logging.getLogger(__name__)

_TRANSLATION_BLOCK_SIZE = 6
_TRANSLATION_BLOCK_OVERLAP = 1
_GLOSSARY_BLOCK_SIZE = 8
_GLOSSARY_BLOCK_OVERLAP = 1


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


def _merge_glossary_terms(existing: list[dict[str, object]], new_terms: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for term in existing + new_terms:
        original = str(term.get("original", "")).strip().lower()
        vietnamese = str(term.get("vietnamese", "")).strip().lower()
        category = str(term.get("category", "term")).strip().lower()
        key = (original, vietnamese, category)
        current = merged.get(key)
        source_segments = list({str(segment) for segment in term.get("source_segments", [])})
        if current is None:
            merged[key] = {
                "term_id": str(term.get("term_id", f"term_{len(merged) + 1:04d}")),
                "original": str(term.get("original", "")),
                "vietnamese": str(term.get("vietnamese", "")),
                "category": str(term.get("category", "term")),
                "confidence": float(term.get("confidence", 1.0)),
                "locked": bool(term.get("locked", True)),
                "source_segments": source_segments,
                "notes": str(term.get("notes", "")),
            }
            continue
        current["confidence"] = max(float(current.get("confidence", 1.0)), float(term.get("confidence", 1.0)))
        current["source_segments"] = sorted({*map(str, current.get("source_segments", [])), *source_segments})
        if not current.get("notes") and term.get("notes"):
            current["notes"] = str(term.get("notes", ""))
    return list(merged.values())


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
    segments = ctx.artifact_json("segments.v1.json")["segments"]
    logger.info(
        "stage asr started segments=%s provider_mode=%s source_audio=%s",
        len(segments),
        ctx.provider_mode,
        ctx.paths.to_relative(ctx.paths.audio_dir / "vocals.wav"),
    )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.ASR, StageStatus.RUNNING, done=0, total=len(segments))
    ctx.store.save()
    segment_store = SegmentCheckpointStore.create(
        ctx.paths.artifact_path("asr_segments.v1.json"),
        stage=StageName.ASR.value,
        segment_ids=[str(segment["segment_id"]) for segment in segments],
    )
    transcript_path = ctx.paths.artifact_path("transcript.v1.json")
    transcript_segments: list[dict[str, object]] = []
    asr_chunks: list[dict[str, object]] = []
    source_audio = ctx.paths.audio_dir / "vocals.wav"
    provider_bundle = ctx.require_provider_bundle() if ctx.provider_mode == "openai_compatible" else None
    stage_started = perf_counter()
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment["segment_id"])
        raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
        if ctx.provider_mode == "openai_compatible":
            assert provider_bundle is not None
            clip_path = ctx.paths.audio_dir / "asr" / f"{segment_id}.wav"
            start_ms = int(segment["start_ms"])
            end_ms = int(segment["end_ms"])
            duration_ms = end_ms - start_ms
            logger.info(
                "stage asr clip prepare segment=%s index=%s/%s start_ms=%s end_ms=%s duration_ms=%s output=%s",
                segment_id,
                index,
                len(segments),
                start_ms,
                end_ms,
                duration_ms,
                ctx.paths.to_relative(clip_path),
            )
            ctx.ffmpeg.extract_audio_segment(
                source_audio,
                clip_path,
                start_ms=start_ms,
                duration_ms=duration_ms,
            )
            clip_size = clip_path.stat().st_size if clip_path.exists() else -1
            logger.info(
                "stage asr request start segment=%s index=%s/%s clip=%s clip_ms=%s clip_bytes=%s provider=%s model=%s",
                segment_id,
                index,
                len(segments),
                ctx.paths.to_relative(clip_path),
                duration_ms,
                clip_size,
                ctx.provider_mode,
                getattr(provider_bundle.asr, "model", "unknown") if provider_bundle is not None else "mock-asr",
            )
            request_started = perf_counter()
            logger.info("stage asr provider call started segment=%s index=%s/%s", segment_id, index, len(segments))
            try:
                asr_result = asyncio.run(
                    provider_bundle.asr.transcribe(
                        clip_path,
                        language="en",
                    )
                )
            except Exception:
                logger.exception(
                    "stage asr provider call failed segment=%s index=%s/%s clip=%s clip_ms=%s raw=%s",
                    segment_id,
                    index,
                    len(segments),
                    ctx.paths.to_relative(clip_path),
                    duration_ms,
                    ctx.paths.to_relative(raw_path),
                )
                raise
            request_duration_ms = int((perf_counter() - request_started) * 1000)
            text = asr_result.text
            confidence = asr_result.confidence if asr_result.confidence is not None else 1.0
            write_json_atomic(raw_path, asr_result.raw)
            normalized_timestamps = normalize_asr_timestamps(
                asr_result.raw,
                chunk_start_ms=start_ms,
                chunk_end_ms=end_ms,
                require_timestamps=ctx.config.asr_service.require_timestamps,
                allow_chunk_text_fallback=ctx.config.asr_service.allow_chunk_text_fallback,
            )
            asr_chunks.append(
                {
                    "chunk_id": segment_id,
                    "raw_response_path": ctx.paths.to_relative(raw_path),
                    "timestamps": normalized_timestamps,
                    "confidence": confidence,
                }
            )
            logger.info(
                "stage asr provider call completed segment=%s index=%s/%s request_duration_ms=%s text_chars=%s confidence=%s language=%s timestamp_source=%s timestamp_units=%s raw=%s",
                segment_id,
                index,
                len(segments),
                request_duration_ms,
                len(text),
                confidence,
                asr_result.language,
                normalized_timestamps.source,
                len(normalized_timestamps.units),
                ctx.paths.to_relative(raw_path),
            )
        else:
            text = f"Mock transcript segment {index} for vertical slice."
            confidence = 1.0
            write_json_atomic(raw_path, {"text": text, "confidence": confidence})
            normalized_timestamps = NormalizedASRTimestamps(
                source="mock",
                quality="mock",
                units=[TimestampUnit(text=text, start_ms=int(segment["start_ms"]), end_ms=int(segment["end_ms"]))],
                risk_flags=[],
            )
            asr_chunks.append(
                {
                    "chunk_id": segment_id,
                    "raw_response_path": ctx.paths.to_relative(raw_path),
                    "timestamps": normalized_timestamps,
                    "confidence": confidence,
                }
            )
        elapsed_ms = int((perf_counter() - stage_started) * 1000)
        average_ms = elapsed_ms / index
        eta_ms = int(average_ms * max(0, len(segments) - index))
        logger.info(
            "stage asr segment completed segment=%s index=%s/%s text_chars=%s raw=%s progress=%.1f%% elapsed_ms=%s avg_segment_ms=%s eta_ms=%s",
            segment_id,
            index,
            len(segments),
            len(text),
            ctx.paths.to_relative(raw_path),
            (index / len(segments)) * 100 if segments else 100.0,
            elapsed_ms,
            int(average_ms),
            eta_ms,
        )
        segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
        segment_store.save()
    transcript_segments = build_transcript_segments(asr_chunks, ctx.config.transcript_segmentation)
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

    logger.info("stage asr completed chunks=%s transcript_segments=%s artifact=artifacts/transcript.v1.json", len(segments), len(transcript_segments))


def run_glossary(ctx: StageContext, *, glossary_review: bool) -> bool:
    domain = _effective_domain(ctx)
    logger.info("stage glossary started review=%s provider_mode=%s domain=%s", glossary_review, ctx.provider_mode, domain)
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.GLOSSARY, StageStatus.RUNNING)
    ctx.store.save()
    glossary_path = ctx.paths.artifact_path("glossary.draft.json" if glossary_review else "glossary.locked.json")
    source_segments = [segment["segment_id"] for segment in ctx.artifact_json("segments.v1.json")["segments"]]
    if ctx.provider_mode == "openai_compatible":
        transcript = ctx.artifact_json("transcript.v1.json")
        blocks = build_translation_blocks(transcript["segments"], block_size=_GLOSSARY_BLOCK_SIZE, overlap=_GLOSSARY_BLOCK_OVERLAP)
        logger.info("stage glossary llm request started blocks=%s segments=%s domain=%s", len(blocks), len(transcript["segments"]), domain)
        terms: list[dict[str, object]] = []
        for block_index, block in enumerate(blocks, start=1):
            block_segment_ids = [str(segment["segment_id"]) for segment in block]
            logger.info("stage glossary block started block=%s/%s segments=%s segment_ids=%s", block_index, len(blocks), len(block), ",".join(block_segment_ids))
            try:
                glossary_result = asyncio.run(
                    _request_structured_json(
                        ctx.require_provider_bundle().llm,
                        f"Extract glossary terms for a {domain} video transcript block. Keep terminology consistent with the {domain} domain. Return raw JSON only.",
                        json.dumps(
                            {
                                "instruction": "Return only a JSON object with a terms array. Do not write prose, markdown, headings, or bullets.",
                                "domain": domain,
                                "segment_window": {
                                    "first_segment_id": block_segment_ids[0] if block_segment_ids else None,
                                    "last_segment_id": block_segment_ids[-1] if block_segment_ids else None,
                                    "segment_count": len(block),
                                    "overlap_segments": _GLOSSARY_BLOCK_OVERLAP,
                                },
                                "segments": block,
                                "required_output_example": {
                                    "terms": [
                                        {
                                            "term_id": "term_01_0001",
                                            "original": "Large language model",
                                            "vietnamese": "mô hình ngôn ngữ lớn",
                                            "category": "term",
                                            "confidence": 0.9,
                                            "source_segments": block_segment_ids[:1],
                                            "notes": "",
                                        }
                                    ]
                                },
                            },
                            ensure_ascii=False,
                        ),
                        _glossary_response_schema(),
                        response_name="glossary_output",
                        response_description="Glossary terms extracted from the transcript in structured JSON format.",
                    )
                )
                raw_terms = glossary_result.get("terms", [])
            except LLMStructuredOutputError as exc:
                raw_terms = _fallback_glossary_terms_from_text(
                    exc.content,
                    block_index=block_index,
                    block_segment_ids=block_segment_ids or source_segments,
                    glossary_review=glossary_review,
                )
            block_terms = []
            if not isinstance(raw_terms, list):
                logger.warning("stage glossary block returned invalid terms type block=%s/%s type=%s", block_index, len(blocks), type(raw_terms).__name__)
                raw_terms = []
            for index, term in enumerate(raw_terms, start=1):
                normalized_term = _normalize_glossary_term(
                    term,
                    block_index=block_index,
                    term_index=index,
                    block_segment_ids=block_segment_ids or source_segments,
                    glossary_review=glossary_review,
                )
                if normalized_term is not None:
                    block_terms.append(normalized_term)
            terms = _merge_glossary_terms(terms, block_terms)
            logger.info("stage glossary block completed block=%s/%s terms=%s merged_terms=%s", block_index, len(blocks), len(block_terms), len(terms))
    else:
        terms = [
            {
                "term_id": "term_0001",
                "original": "vertical slice",
                "vietnamese": "lát cắt dọc",
                "category": "phrase",
                "locked": not glossary_review,
                "source_segments": source_segments,
                "notes": "Mock glossary entry.",
            }
        ]
    write_json_atomic(
        glossary_path,
        {
            "schema_version": "1.0",
            "domain": domain,
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
        logger.info("stage glossary waiting_review artifact=%s terms=%s", ctx.paths.to_relative(glossary_path), len(terms))
        return True
    publisher.publish_file(
        stage=StageName.GLOSSARY,
        name="glossary",
        path=glossary_path,
    )
    logger.info("stage glossary completed artifact=%s terms=%s", ctx.paths.to_relative(glossary_path), len(terms))
    return False


def run_translation(ctx: StageContext) -> None:
    domain = _effective_domain(ctx)
    logger.info("stage translation started provider_mode=%s domain=%s", ctx.provider_mode, domain)
    transcript = ctx.artifact_json("transcript.v1.json")
    glossary = ctx.artifact_json("glossary.locked.json")
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.TRANSLATION, StageStatus.RUNNING, done=0, total=len(transcript["segments"]))
    ctx.store.save()
    translated_segments: list[dict[str, object]] = []
    provider_translations: dict[str, dict] = {}
    blocks = build_translation_blocks(transcript["segments"], block_size=_TRANSLATION_BLOCK_SIZE, overlap=_TRANSLATION_BLOCK_OVERLAP)
    if ctx.provider_mode == "openai_compatible":
        logger.info("stage translation llm request started blocks=%s segments=%s domain=%s", len(blocks), len(transcript["segments"]), domain)
        for block_index, block in enumerate(blocks, start=1):
            block_segment_ids = [str(segment["segment_id"]) for segment in block]
            logger.info("stage translation block started block=%s/%s segments=%s segment_ids=%s", block_index, len(blocks), len(block), ",".join(block_segment_ids))
            translation_result = asyncio.run(
                _request_structured_json(
                    ctx.require_provider_bundle().llm,
                    f"Translate a {domain} video transcript block to Vietnamese. Preserve the meaning of the {domain} terminology in the glossary and keep the context consistent across overlapping segments. Return raw JSON only.",
                    json.dumps(
                        {
                            "instruction": "Return only a JSON object with a segments array. Do not write prose, markdown, headings, or bullets.",
                            "domain": domain,
                            "segment_window": {
                                "first_segment_id": block_segment_ids[0] if block_segment_ids else None,
                                "last_segment_id": block_segment_ids[-1] if block_segment_ids else None,
                                "segment_count": len(block),
                                "overlap_segments": _TRANSLATION_BLOCK_OVERLAP,
                            },
                            "segments": block,
                            "glossary": glossary["terms"],
                        },
                        ensure_ascii=False,
                    ),
                    _translation_response_schema(),
                    response_name="translation_output",
                    response_description="Vietnamese translation segments in structured JSON format.",
                )
            )
            block_translations = 0
            for segment in translation_result.get("segments", []):
                segment_id = segment.get("segment_id")
                if segment_id is None:
                    continue
                provider_translations[str(segment_id)] = segment
                block_translations += 1
            logger.info("stage translation block completed block=%s/%s translations=%s", block_index, len(blocks), block_translations)
        missing_segments = [str(source["segment_id"]) for source in transcript["segments"] if str(source["segment_id"]) not in provider_translations]
        if missing_segments:
            raise ValueError(f"Translation response missing segment_ids: {', '.join(missing_segments[:5])}")
    for index, source in enumerate(transcript["segments"], start=1):
        logger.info("stage translation segment started segment=%s index=%s/%s", source["segment_id"], index, len(transcript["segments"]))
        if ctx.provider_mode == "openai_compatible":
            provider_segment = provider_translations.get(str(source["segment_id"]), {})
            candidate = {
                "segment_id": source["segment_id"],
                "source_text": source["source_text"],
                "vi_text": provider_segment.get("vi_text", ""),
                "used_terms": provider_segment.get("used_terms", []),
                "length_ratio": provider_segment.get("length_ratio", 1.0),
                "translation_warnings": provider_segment.get("translation_warnings", []),
            }
        else:
            candidate = {
                "segment_id": source["segment_id"],
                "source_text": source["source_text"],
                "vi_text": f"Đây là lát cắt dọc thuyết minh tiếng Việt mẫu cho {source['segment_id']}.",
                "used_terms": ["vertical slice"],
                "length_ratio": 1.0,
                "translation_warnings": [],
            }
        compressed = compress_segment_translation(candidate, glossary["terms"], max_length_ratio=3.0)
        candidate["vi_text"] = compressed.vi_text
        candidate["translation_warnings"] = candidate["translation_warnings"] + compressed.warnings
        translated_segments.append(candidate)
        logger.info("stage translation segment completed segment=%s index=%s/%s chars=%s", source["segment_id"], index, len(transcript["segments"]), len(candidate["vi_text"]))
    validation = validate_translations(
        transcript["segments"],
        translated_segments,
        glossary["terms"],
        max_length_ratio=3.0,
    )
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="translated",
        filename="translated.v1.json",
        payload={
            "schema_version": "1.0",
            "segments": translated_segments,
            "validation_warnings": validation.warnings,
        },
        done=len(transcript["segments"]),
        total=len(transcript["segments"]),
    )

    logger.info("stage translation completed segments=%s warnings=%s artifact=artifacts/translated.v1.json", len(transcript["segments"]), len(validation.warnings))


def run_tts(
    ctx: StageContext,
    duration_ms: int,
    *,
    crash_stage: str | None = None,
    crash_after_segments: int | None = None,
) -> Path:
    transcript_segments = ctx.artifact_json("transcript.v1.json")["segments"]
    segments = transcript_segments
    logger.info("stage tts started segments=%s provider_mode=%s segment_source=transcript", len(segments), ctx.provider_mode)
    ctx.store.mark_stage(StageName.TTS, StageStatus.RUNNING, done=0, total=len(segments))
    ctx.store.save()
    segment_store = SegmentCheckpointStore.create(
        ctx.paths.artifact_path("tts_segments.v1.json"),
        stage=StageName.TTS.value,
        segment_ids=[str(segment["segment_id"]) for segment in segments],
    )
    if crash_stage == StageName.TTS.value and crash_after_segments == 0:
        segment_store.save()
        raise RuntimeError("simulated crash at tts after 0 segments")
    transcript_by_id = {
        str(segment["segment_id"]): segment
        for segment in transcript_segments
    }
    translations_by_id = {
        str(segment["segment_id"]): segment
        for segment in ctx.artifact_json("translated.v1.json")["segments"]
    }
    mix_audio_path = ctx.paths.tts_dir / "mix.wav"
    tts_segments = []
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment["segment_id"])
        logger.info("stage tts segment started segment=%s index=%s/%s", segment_id, index, len(segments))
        if ctx.provider_mode == "openai_compatible":
            source_text = str(transcript_by_id.get(segment_id, {}).get("source_text", ""))
            translated_text = str(translations_by_id.get(segment_id, {}).get("vi_text", ""))
            row = asyncio.run(
                produce_provider_tts_segment(
                    paths=ctx.paths,
                    segment=segment,
                    text="" if not source_text.strip() else translated_text,
                    provider_bundle=ctx.require_provider_bundle(),
                    ffmpeg=ctx.ffmpeg,
                )
            )
        else:
            row = produce_mock_tts_segment(paths=ctx.paths, segment=segment)
        tts_segments.append(row)
        logger.info("stage tts segment completed segment=%s index=%s/%s audio=%s", segment_id, index, len(segments), row["audio_path"])
        segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=row["audio_path"])
        segment_store.save()
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    assemble_commentary_track(
        paths=ctx.paths,
        ffmpeg=ctx.ffmpeg,
        tts_segments=tts_segments,
        output_audio=mix_audio_path,
    )
    logger.info("stage tts commentary track assembled audio=%s segments=%s", ctx.paths.to_relative(mix_audio_path), len(tts_segments))

    publisher.publish_file(
        stage=StageName.TTS,
        name="tts_segments",
        path=segment_store.path,
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
    logger.info("stage tts completed segments=%s mix_audio=%s", len(segments), ctx.paths.to_relative(mix_audio_path))
    return mix_audio_path


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
    ctx.ffmpeg.mux_video_audio(copied_input, final_audio, output_video)
    qa_path = ctx.paths.output_dir / f"{copied_input.stem}_vi.qa.json"
    write_json_atomic(
        qa_path,
        {
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "input_duration_ms": duration_ms,
            "output_duration_ms": ctx.ffmpeg.duration_ms(output_video),
            "segments_total": len(ctx.artifact_json("segments.v1.json")["segments"]),
            "low_confidence_segments": 0,
            "glossary_terms": 1,
            "tts_overflow_segments": 0,
            "max_overflow_ms": 0,
            "sync_drift_p95_ms": 0,
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
    publisher.publish_file(
        stage=StageName.MIXING,
        name="output_video",
        path=output_video,
        status=StageStatus.COMPLETED,
    )
    logger.info("stage mixing completed output=%s qa=%s", output_video, qa_path)
    return output_video
