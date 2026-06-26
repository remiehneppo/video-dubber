from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from dubber.audio.vad import VadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations


def run_job_init(ctx: StageContext, copied_input: Path) -> None:
    ctx.store.mark_stage(StageName.JOB_INIT, StageStatus.RUNNING)
    ctx.store.save()
    metadata_path = ctx.paths.input_dir / "input_metadata.v1.json"
    write_json_atomic(metadata_path, ctx.ffmpeg.probe(copied_input))
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
        stage=StageName.JOB_INIT,
        name="input_metadata",
        path=metadata_path,
    )


def run_audio_extract(ctx: StageContext, copied_input: Path) -> int:
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
    return duration_ms


def run_vad(ctx: StageContext) -> None:
    ctx.store.mark_stage(StageName.VAD, StageStatus.RUNNING)
    ctx.store.save()
    segments = detect_segments(
        ctx.paths.audio_dir / "vocals.wav",
        VadConfig(
            min_duration_ms=300,
            max_duration_ms=25_000,
            silence_merge_threshold_ms=400,
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


def run_asr(ctx: StageContext) -> None:
    segments = ctx.artifact_json("segments.v1.json")["segments"]
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
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment["segment_id"])
        raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
        if ctx.provider_mode == "openai_compatible":
            asr_result = asyncio.run(
                ctx.require_provider_bundle().asr.transcribe(
                    ctx.paths.audio_dir / "vocals.wav",
                    language="en",
                )
            )
            text = asr_result.text
            confidence = asr_result.confidence if asr_result.confidence is not None else 1.0
            write_json_atomic(raw_path, asr_result.raw)
        else:
            text = f"Mock transcript segment {index} for vertical slice."
            confidence = 1.0
            write_json_atomic(raw_path, {"text": text, "confidence": confidence})
        segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
        segment_store.save()
        transcript_segments.append(
            {
                "segment_id": segment_id,
                "start_ms": int(segment["start_ms"]),
                "end_ms": int(segment["end_ms"]),
                "source_text": text,
                "confidence": confidence,
                "asr_warnings": [],
                "raw_response_path": ctx.paths.to_relative(raw_path),
            }
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


def run_glossary(ctx: StageContext, *, glossary_review: bool) -> bool:
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.GLOSSARY, StageStatus.RUNNING)
    ctx.store.save()
    glossary_path = ctx.paths.artifact_path("glossary.draft.json" if glossary_review else "glossary.locked.json")
    source_segments = [segment["segment_id"] for segment in ctx.artifact_json("segments.v1.json")["segments"]]
    if ctx.provider_mode == "openai_compatible":
        transcript = ctx.artifact_json("transcript.v1.json")
        glossary_result = asyncio.run(
            ctx.require_provider_bundle().llm.complete_json(
                "extract glossary",
                "terminology " + str(transcript["segments"]),
                schema={"type": "object"},
            )
        )
        terms = []
        for index, term in enumerate(glossary_result.get("terms", []), start=1):
            terms.append(
                {
                    "term_id": term.get("term_id", f"term_{index:04d}"),
                    "original": term.get("original", ""),
                    "vietnamese": term.get("vietnamese", ""),
                    "category": term.get("category", "term"),
                    "confidence": term.get("confidence", 1.0),
                    "locked": not glossary_review,
                    "source_segments": term.get("source_segments", source_segments),
                    "notes": term.get("notes", ""),
                }
            )
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
            "domain": ctx.provider_mode,
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
    publisher.publish_file(
        stage=StageName.GLOSSARY,
        name="glossary",
        path=glossary_path,
    )
    return False


def run_translation(ctx: StageContext) -> None:
    transcript = ctx.artifact_json("transcript.v1.json")
    glossary = ctx.artifact_json("glossary.locked.json")
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.TRANSLATION, StageStatus.RUNNING, done=0, total=len(transcript["segments"]))
    ctx.store.save()
    translated_segments = []
    provider_translations: dict[str, dict] = {}
    if ctx.provider_mode == "openai_compatible":
        translation_result = asyncio.run(
            ctx.require_provider_bundle().llm.complete_json(
                "translate transcript to Vietnamese",
                str({"segments": transcript["segments"], "glossary": glossary["terms"]}),
                schema={"type": "object"},
            )
        )
        provider_translations = {
            str(segment.get("segment_id")): segment
            for segment in translation_result.get("segments", [])
            if segment.get("segment_id") is not None
        }
    for source in transcript["segments"]:
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
