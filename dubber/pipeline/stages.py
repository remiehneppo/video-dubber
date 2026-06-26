from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from dubber.audio.vad import VadConfig as AudioVadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext
from dubber.tts.mock import synthesize_tone_wav
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_segment
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations

logger = logging.getLogger(__name__)


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
    logger.info("stage vad started frame_ms=%s threshold_ratio=%s min_duration_ms=%s max_duration_ms=%s", ctx.config.vad.frame_ms, ctx.config.vad.threshold_ratio, ctx.config.vad.min_duration_ms, ctx.config.vad.max_duration_ms)
    ctx.store.mark_stage(StageName.VAD, StageStatus.RUNNING)
    ctx.store.save()
    vad = ctx.config.vad
    segments = detect_segments(
        ctx.paths.audio_dir / "vocals.wav",
        AudioVadConfig(
            frame_ms=vad.frame_ms,
            threshold_ratio=vad.threshold_ratio,
            min_duration_ms=vad.min_duration_ms,
            max_duration_ms=vad.max_duration_ms,
            silence_merge_threshold_ms=vad.silence_merge_threshold_ms,
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
    logger.info("stage asr started segments=%s provider_mode=%s", len(segments), ctx.provider_mode)
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
            logger.info("stage asr request segment=%s index=%s/%s", segment_id, index, len(segments))
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
        logger.info("stage asr segment completed segment=%s index=%s/%s text_chars=%s", segment_id, index, len(segments), len(text))
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

    logger.info("stage asr completed segments=%s artifact=artifacts/transcript.v1.json", len(segments))


def run_glossary(ctx: StageContext, *, glossary_review: bool) -> bool:
    logger.info("stage glossary started review=%s provider_mode=%s", glossary_review, ctx.provider_mode)
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.GLOSSARY, StageStatus.RUNNING)
    ctx.store.save()
    glossary_path = ctx.paths.artifact_path("glossary.draft.json" if glossary_review else "glossary.locked.json")
    source_segments = [segment["segment_id"] for segment in ctx.artifact_json("segments.v1.json")["segments"]]
    if ctx.provider_mode == "openai_compatible":
        logger.info("stage glossary llm request started")
        transcript = ctx.artifact_json("transcript.v1.json")
        glossary_result = asyncio.run(
            ctx.require_provider_bundle().llm.complete_json(
                "extract glossary",
                "terminology " + str(transcript["segments"]),
                schema={"type": "object"},
            )
        )
        logger.info("stage glossary llm request completed terms=%s", len(glossary_result.get("terms", [])))
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
    logger.info("stage translation started provider_mode=%s", ctx.provider_mode)
    transcript = ctx.artifact_json("transcript.v1.json")
    glossary = ctx.artifact_json("glossary.locked.json")
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    ctx.store.mark_stage(StageName.TRANSLATION, StageStatus.RUNNING, done=0, total=len(transcript["segments"]))
    ctx.store.save()
    translated_segments = []
    provider_translations: dict[str, dict] = {}
    if ctx.provider_mode == "openai_compatible":
        logger.info("stage translation llm request started segments=%s", len(transcript["segments"]))
        translation_result = asyncio.run(
            ctx.require_provider_bundle().llm.complete_json(
                "translate transcript to Vietnamese",
                str({"segments": transcript["segments"], "glossary": glossary["terms"]}),
                schema={"type": "object"},
            )
        )
        logger.info("stage translation llm request completed")
        provider_translations = {
            str(segment.get("segment_id")): segment
            for segment in translation_result.get("segments", [])
            if segment.get("segment_id") is not None
        }
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
    segments = ctx.artifact_json("segments.v1.json")["segments"]
    logger.info("stage tts started segments=%s provider_mode=%s", len(segments), ctx.provider_mode)
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
    translations_by_id = {
        str(segment["segment_id"]): segment
        for segment in ctx.artifact_json("translated.v1.json")["segments"]
    }
    mix_audio_path = ctx.paths.tts_dir / "mix.wav"
    synthesize_tone_wav(mix_audio_path, duration_ms)
    tts_segments = []
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment["segment_id"])
        logger.info("stage tts segment started segment=%s index=%s/%s", segment_id, index, len(segments))
        if ctx.provider_mode == "openai_compatible":
            row = asyncio.run(
                produce_provider_tts_segment(
                    paths=ctx.paths,
                    segment=segment,
                    text=translations_by_id.get(segment_id, {}).get("vi_text", ""),
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
    ctx.ffmpeg.mix_commentary_audio(ctx.paths.audio_dir / "original.wav", tts_audio, final_audio)
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
