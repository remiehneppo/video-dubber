from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from dubber.core.config import load_config
from dubber.core.enums import JobStatus, StageName, StageStatus
from dubber.core.io import read_json, write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.audio.vad import VadConfig, detect_segments
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import run_asr, run_audio_extract, run_glossary, run_job_init, run_translation, run_vad
from dubber.providers.factory import ProviderBundle, build_provider_bundle
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.aligner import apply_time_stretch
from dubber.tts.duration_planner import plan_segment_duration
from dubber.tts.mock import synthesize_tone_wav
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations


@dataclass(frozen=True)
class RunOptions:
    input_path: Path
    workspace_dir: Path
    config_path: Path | None = None
    domain: str | None = None
    provider_mode: str = "mock"
    glossary_review: bool = False
    crash_stage: str | None = None
    crash_after_segments: int | None = None


@dataclass(frozen=True)
class RunSummary:
    job_id: str
    status: str
    output_video: str
    workspace: str

    def to_dict(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "output_video": self.output_video,
            "workspace": self.workspace,
        }


class JobManager:
    def __init__(self, ffmpeg: FFmpegAdapter | None = None, provider_bundle: ProviderBundle | None = None) -> None:
        self.ffmpeg = ffmpeg or FFmpegAdapter()
        self.provider_bundle = provider_bundle
        self.provider_mode = "mock"

    def run(self, options: RunOptions) -> RunSummary:
        config = load_config(options.config_path or Path("config.example.yaml"))
        if options.provider_mode not in {"mock", "openai_compatible"}:
            raise ValueError(f"Unsupported provider-mode: {options.provider_mode}")
        self.provider_mode = options.provider_mode
        if options.provider_mode == "openai_compatible" and self.provider_bundle is None:
            self.provider_bundle = build_provider_bundle(config)

        input_path = options.input_path.expanduser().resolve()
        self._validate_input(input_path, config.input.allowed_extensions, config.input.max_file_size_mb)

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        paths = WorkspacePaths.create(options.workspace_dir, job_id)
        store = CheckpointStore.create(paths.job_state_file, job_id=job_id, input_file=Path("input") / input_path.name)
        manifest = ArtifactManifest.create(job_id, paths.manifest_file)
        ctx = StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
        )

        copied_input = paths.input_dir / input_path.name
        shutil.copy2(input_path, copied_input)
        self._write_resolved_config(paths, options, config.project.domain)

        try:
            run_job_init(ctx, copied_input)
            duration_ms = run_audio_extract(ctx, copied_input)
            run_vad(ctx)
            run_asr(ctx)
            if run_glossary(ctx, glossary_review=options.glossary_review):
                store.mark_job(JobStatus.WAITING_REVIEW)
                store.save()
                manifest.save()
                return RunSummary(
                    job_id=job_id,
                    status=JobStatus.WAITING_REVIEW.value,
                    output_video="",
                    workspace=str(paths.root),
                )
            run_translation(ctx)
            tts_audio = self._stage_tts(duration_ms, paths, store, manifest, crash_stage=options.crash_stage, crash_after_segments=options.crash_after_segments)
            output_video = self._stage_mixing(copied_input, tts_audio, duration_ms, paths, store, manifest)
        except Exception as exc:
            store.mark_job(JobStatus.FAILED, error=str(exc))
            store.save()
            raise

        store.mark_job(JobStatus.COMPLETED)
        store.save()
        manifest.save()
        return RunSummary(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            output_video=paths.to_relative(output_video),
            workspace=str(paths.root),
        )

    def resume(self, workspace_dir: Path, job_id: str) -> RunSummary:
        paths = WorkspacePaths.create(workspace_dir, job_id)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        ctx = StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
        )
        if store.state.status in {JobStatus.FAILED, JobStatus.RUNNING} and store.state.current_stage == StageName.TTS:
            copied_input = paths.resolve_relative(store.state.input_file)
            duration_ms = self.ffmpeg.duration_ms(copied_input)
            store.mark_job(JobStatus.RUNNING)
            store.save()
            tts_audio = self._stage_tts(duration_ms, paths, store, manifest)
            output_video = self._stage_mixing(copied_input, tts_audio, duration_ms, paths, store, manifest)
            store.mark_job(JobStatus.COMPLETED)
            store.save()
            manifest.save()
            return RunSummary(
                job_id=job_id,
                status=JobStatus.COMPLETED.value,
                output_video=paths.to_relative(output_video),
                workspace=str(paths.root),
            )

        if store.state.status != JobStatus.WAITING_REVIEW:
            raise ValueError(f"Job {job_id} is not waiting for glossary review or resumable stage")

        locked_glossary = paths.artifact_path("glossary.locked.json")
        if not locked_glossary.exists():
            raise FileNotFoundError(
                f"Reviewed glossary is missing: {paths.to_relative(locked_glossary)}"
            )
        manifest.record_artifact(
            name="glossary",
            version=1,
            path=locked_glossary,
            created_by_stage=StageName.GLOSSARY,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(StageName.GLOSSARY, StageStatus.COMPLETED, artifact=paths.to_relative(locked_glossary))
        store.mark_job(JobStatus.RUNNING)
        store.save()

        copied_input = paths.resolve_relative(store.state.input_file)
        duration_ms = self.ffmpeg.duration_ms(copied_input)
        run_translation(ctx)
        tts_audio = self._stage_tts(duration_ms, paths, store, manifest)
        output_video = self._stage_mixing(copied_input, tts_audio, duration_ms, paths, store, manifest)
        store.mark_job(JobStatus.COMPLETED)
        store.save()
        manifest.save()
        return RunSummary(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            output_video=paths.to_relative(output_video),
            workspace=str(paths.root),
        )

    def rerun_stage(self, workspace_dir: Path, job_id: str, stage: StageName) -> RunSummary:
        paths = WorkspacePaths.create(workspace_dir, job_id)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        ctx = StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
        )
        copied_input = paths.resolve_relative(store.state.input_file)
        duration_ms = self.ffmpeg.duration_ms(copied_input)

        store.mark_job(JobStatus.RUNNING)
        store.save()
        if stage == StageName.TRANSLATION:
            run_translation(ctx)
            tts_audio = self._stage_tts(duration_ms, paths, store, manifest)
            output_video = self._stage_mixing(copied_input, tts_audio, duration_ms, paths, store, manifest)
        elif stage == StageName.TTS:
            tts_audio = self._stage_tts(duration_ms, paths, store, manifest)
            output_video = self._stage_mixing(copied_input, tts_audio, duration_ms, paths, store, manifest)
        elif stage == StageName.MIXING:
            tts_manifest = read_json(paths.artifact_path("tts_manifest.v1.json"))
            first_audio = paths.resolve_relative(tts_manifest["segments"][0]["audio_path"])
            output_video = self._stage_mixing(copied_input, first_audio, duration_ms, paths, store, manifest)
        else:
            raise ValueError(f"Rerun is not supported for stage: {stage.value}")

        store.mark_job(JobStatus.COMPLETED)
        store.save()
        manifest.save()
        return RunSummary(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            output_video=paths.to_relative(output_video),
            workspace=str(paths.root),
        )

    def rerun_segment(self, workspace_dir: Path, job_id: str, stage: StageName, segment_id: str) -> RunSummary:
        if stage != StageName.TTS:
            raise ValueError(f"Segment rerun is not supported for stage: {stage.value}")
        paths = WorkspacePaths.create(workspace_dir, job_id)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        copied_input = paths.resolve_relative(store.state.input_file)
        duration_ms = self.ffmpeg.duration_ms(copied_input)
        segments = read_json(paths.artifact_path("segments.v1.json"))["segments"]
        matching = [segment for segment in segments if str(segment["segment_id"]) == segment_id]
        if not matching:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        segment = matching[0]

        checkpoint = SegmentCheckpointStore.load(paths.artifact_path("tts_segments.v1.json"))
        raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
        audio_path = paths.tts_dir / f"{segment_id}.wav"
        orig_ms = int(segment["duration_ms"])
        mock_tts_ms = max(100, int(orig_ms * 1.1))
        synthesize_tone_wav(raw_audio_path, mock_tts_ms)
        timing_plan = plan_segment_duration(segment_id, orig_duration_ms=orig_ms, tts_duration_ms=mock_tts_ms)
        if timing_plan.action == "time_stretch":
            apply_time_stretch(raw_audio_path, audio_path, timing_plan.stretch_ratio)
        else:
            apply_time_stretch(raw_audio_path, audio_path, 1.0)
        checkpoint.mark(segment_id, StageStatus.COMPLETED, artifact=paths.to_relative(audio_path))
        checkpoint.save()

        tts_manifest_path = paths.artifact_path("tts_manifest.v1.json")
        tts_manifest = read_json(tts_manifest_path)
        for item in tts_manifest["segments"]:
            if str(item["segment_id"]) == segment_id:
                item["tts_duration_ms"] = mock_tts_ms
                item["alignment_action"] = timing_plan.action
                item["stretch_ratio"] = timing_plan.stretch_ratio
                item["overflow_ms"] = timing_plan.overflow_ms
                item["raw_audio_path"] = paths.to_relative(raw_audio_path)
                item["audio_path"] = paths.to_relative(audio_path)
                item["warnings"] = timing_plan.warnings
        write_json_atomic(tts_manifest_path, tts_manifest)
        manifest.record_artifact(name="tts_segments", version=1, path=checkpoint.path, created_by_stage=StageName.TTS, schema_version="1.0")
        manifest.record_artifact(name="tts_manifest", version=1, path=tts_manifest_path, created_by_stage=StageName.TTS, schema_version="1.0")
        output_video = self._stage_mixing(copied_input, audio_path, duration_ms, paths, store, manifest)
        store.mark_stage(StageName.TTS, StageStatus.COMPLETED, artifact=paths.to_relative(tts_manifest_path), done=checkpoint.done_count, total=checkpoint.total_count)
        store.mark_job(JobStatus.COMPLETED)
        store.save()
        manifest.save()
        return RunSummary(job_id=job_id, status=JobStatus.COMPLETED.value, output_video=paths.to_relative(output_video), workspace=str(paths.root))

    def _validate_input(self, input_path: Path, allowed_extensions: list[str], max_file_size_mb: int) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        if input_path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported input extension: {input_path.suffix}")
        if input_path.stat().st_size > max_file_size_mb * 1024 * 1024:
            raise ValueError(f"Input file exceeds max size: {max_file_size_mb} MB")
        if not self.ffmpeg.has_audio_stream(input_path):
            raise ValueError("Input video has no audio stream")

    def _write_resolved_config(self, paths: WorkspacePaths, options: RunOptions, default_domain: str) -> None:
        write_json_atomic(
            paths.root / "config.resolved.json",
            {
                "provider_mode": options.provider_mode,
                "domain": options.domain or default_domain,
                "glossary_review": options.glossary_review,
            },
        )

    def _stage_job_init(
        self,
        copied_input: Path,
        paths: WorkspacePaths,
        store: CheckpointStore,
        manifest: ArtifactManifest,
    ) -> None:
        store.mark_stage(StageName.JOB_INIT, StageStatus.RUNNING)
        store.save()
        metadata_path = paths.input_dir / "input_metadata.v1.json"
        write_json_atomic(metadata_path, self.ffmpeg.probe(copied_input))
        manifest.record_artifact(
            name="input_metadata",
            version=1,
            path=metadata_path,
            created_by_stage=StageName.JOB_INIT,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(StageName.JOB_INIT, StageStatus.COMPLETED, artifact=paths.to_relative(metadata_path))
        store.save()

    def _stage_audio_extract(
        self,
        copied_input: Path,
        paths: WorkspacePaths,
        store: CheckpointStore,
        manifest: ArtifactManifest,
    ) -> int:
        store.mark_stage(StageName.AUDIO_EXTRACT, StageStatus.RUNNING)
        store.save()
        original_wav = paths.audio_dir / "original.wav"
        self.ffmpeg.extract_audio(copied_input, original_wav)
        shutil.copy2(original_wav, paths.audio_dir / "vocals.wav")
        duration_ms = self.ffmpeg.duration_ms(copied_input)
        analysis_path = paths.artifact_path("audio_analysis.v1.json")
        write_json_atomic(
            analysis_path,
            {
                "schema_version": "1.0",
                "audio_duration_ms": duration_ms,
                "sample_rate": 44100,
                "channels": 1,
                "source_separation_used": False,
                "source_separation_reason": "mock_vertical_slice",
            },
        )
        manifest.record_artifact(
            name="audio_analysis",
            version=1,
            path=analysis_path,
            created_by_stage=StageName.AUDIO_EXTRACT,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(StageName.AUDIO_EXTRACT, StageStatus.COMPLETED, artifact=paths.to_relative(analysis_path))
        store.save()
        return duration_ms

    def _stage_vad(self, paths: WorkspacePaths, store: CheckpointStore, manifest: ArtifactManifest) -> None:
        store.mark_stage(StageName.VAD, StageStatus.RUNNING)
        store.save()
        segments_path = paths.artifact_path("segments.v1.json")
        segments = detect_segments(
            paths.audio_dir / "vocals.wav",
            VadConfig(
                min_duration_ms=300,
                max_duration_ms=25_000,
                silence_merge_threshold_ms=400,
            ),
        )
        write_json_atomic(
            segments_path,
            {
                "schema_version": "1.0",
                "job_id": paths.root.name,
                "source_audio": "audio/vocals.wav",
                "segments": [segment.to_dict() for segment in segments],
            },
        )
        manifest.record_artifact(
            name="segments",
            version=1,
            path=segments_path,
            created_by_stage=StageName.VAD,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(StageName.VAD, StageStatus.COMPLETED, artifact=paths.to_relative(segments_path))
        store.save()

    def _stage_asr(self, paths: WorkspacePaths, store: CheckpointStore, manifest: ArtifactManifest) -> None:
        segments = read_json(paths.artifact_path("segments.v1.json"))["segments"]
        store.mark_stage(StageName.ASR, StageStatus.RUNNING, done=0, total=len(segments))
        store.save()
        segment_store = SegmentCheckpointStore.create(
            paths.artifact_path("asr_segments.v1.json"),
            stage=StageName.ASR.value,
            segment_ids=[str(segment["segment_id"]) for segment in segments],
        )
        transcript_path = paths.artifact_path("transcript.v1.json")
        transcript_segments = []
        for index, segment in enumerate(segments, start=1):
            segment_id = str(segment["segment_id"])
            raw_path = paths.raw_dir / "asr" / f"{segment_id}.json"
            if self.provider_mode == "openai_compatible":
                if self.provider_bundle is None:
                    raise RuntimeError("provider bundle is not configured")
                asr_result = asyncio.run(self.provider_bundle.asr.transcribe(paths.audio_dir / "vocals.wav", language="en"))
                text = asr_result.text
                confidence = asr_result.confidence if asr_result.confidence is not None else 1.0
                write_json_atomic(raw_path, asr_result.raw)
            else:
                text = f"Mock transcript segment {index} for vertical slice."
                confidence = 1.0
                write_json_atomic(raw_path, {"text": text, "confidence": confidence})
            segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=paths.to_relative(raw_path))
            segment_store.save()
            transcript_segments.append(
                {
                    "segment_id": segment_id,
                    "start_ms": int(segment["start_ms"]),
                    "end_ms": int(segment["end_ms"]),
                    "source_text": text,
                    "confidence": confidence,
                    "asr_warnings": [],
                    "raw_response_path": paths.to_relative(raw_path),
                }
            )
        write_json_atomic(
            transcript_path,
            {
                "schema_version": "1.0",
                "provider": {"type": self.provider_mode, "model": "provider-asr" if self.provider_mode == "openai_compatible" else "mock-asr"},
                "segments": transcript_segments,
            },
        )
        manifest.record_artifact(
            name="asr_segments",
            version=1,
            path=segment_store.path,
            created_by_stage=StageName.ASR,
            schema_version="1.0",
        )
        manifest.record_artifact(
            name="transcript",
            version=1,
            path=transcript_path,
            created_by_stage=StageName.ASR,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(
            StageName.ASR,
            StageStatus.COMPLETED,
            artifact=paths.to_relative(transcript_path),
            done=len(segments),
            total=len(segments),
        )
        store.save()

    def _stage_glossary(
        self,
        paths: WorkspacePaths,
        store: CheckpointStore,
        manifest: ArtifactManifest,
        *,
        glossary_review: bool,
    ) -> bool:
        store.mark_stage(StageName.GLOSSARY, StageStatus.RUNNING)
        store.save()
        glossary_path = paths.artifact_path("glossary.draft.json" if glossary_review else "glossary.locked.json")
        source_segments = [segment["segment_id"] for segment in read_json(paths.artifact_path("segments.v1.json"))["segments"]]
        if self.provider_mode == "openai_compatible":
            if self.provider_bundle is None:
                raise RuntimeError("provider bundle is not configured")
            transcript = read_json(paths.artifact_path("transcript.v1.json"))
            glossary_result = asyncio.run(self.provider_bundle.llm.complete_json("extract glossary", "terminology " + str(transcript["segments"]), schema={"type": "object"}))
            terms = []
            for index, term in enumerate(glossary_result.get("terms", []), start=1):
                terms.append({
                    "term_id": term.get("term_id", f"term_{index:04d}"),
                    "original": term.get("original", ""),
                    "vietnamese": term.get("vietnamese", ""),
                    "category": term.get("category", "term"),
                    "confidence": term.get("confidence", 1.0),
                    "locked": not glossary_review,
                    "source_segments": term.get("source_segments", source_segments),
                    "notes": term.get("notes", ""),
                })
        else:
            terms = [{
                "term_id": "term_0001",
                "original": "vertical slice",
                "vietnamese": "lát cắt dọc",
                "category": "phrase",
                "locked": not glossary_review,
                "source_segments": source_segments,
                "notes": "Mock glossary entry.",
            }]
        write_json_atomic(glossary_path, {"schema_version": "1.0", "domain": self.provider_mode, "status": "draft" if glossary_review else "locked", "terms": terms})
        if glossary_review:
            manifest.record_artifact(
                name="glossary_draft",
                version=1,
                path=glossary_path,
                created_by_stage=StageName.GLOSSARY,
                schema_version="1.0",
            )
            manifest.save()
            store.mark_stage(
                StageName.GLOSSARY,
                StageStatus.WAITING_REVIEW,
                artifact=paths.to_relative(glossary_path),
            )
            store.save()
            return True
        manifest.record_artifact(
            name="glossary",
            version=1,
            path=glossary_path,
            created_by_stage=StageName.GLOSSARY,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(StageName.GLOSSARY, StageStatus.COMPLETED, artifact=paths.to_relative(glossary_path))
        store.save()
        return False

    def _stage_translation(self, paths: WorkspacePaths, store: CheckpointStore, manifest: ArtifactManifest) -> None:
        transcript = read_json(paths.artifact_path("transcript.v1.json"))
        store.mark_stage(StageName.TRANSLATION, StageStatus.RUNNING, done=0, total=len(transcript["segments"]))
        store.save()
        translated_path = paths.artifact_path("translated.v1.json")
        glossary = read_json(paths.artifact_path("glossary.locked.json"))
        translated_segments = []
        provider_translations: dict[str, dict] = {}
        if self.provider_mode == "openai_compatible":
            if self.provider_bundle is None:
                raise RuntimeError("provider bundle is not configured")
            translation_result = asyncio.run(
                self.provider_bundle.llm.complete_json(
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
            if self.provider_mode == "openai_compatible":
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
        write_json_atomic(
            translated_path,
            {
                "schema_version": "1.0",
                "segments": translated_segments,
                "validation_warnings": validation.warnings,
            },
        )
        manifest.record_artifact(
            name="translated",
            version=1,
            path=translated_path,
            created_by_stage=StageName.TRANSLATION,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(
            StageName.TRANSLATION,
            StageStatus.COMPLETED,
            artifact=paths.to_relative(translated_path),
            done=len(transcript["segments"]),
            total=len(transcript["segments"]),
        )
        store.save()

    def _stage_tts(
        self,
        duration_ms: int,
        paths: WorkspacePaths,
        store: CheckpointStore,
        manifest: ArtifactManifest,
        *,
        crash_stage: str | None = None,
        crash_after_segments: int | None = None,
    ) -> Path:
        segments = read_json(paths.artifact_path("segments.v1.json"))["segments"]
        store.mark_stage(StageName.TTS, StageStatus.RUNNING, done=0, total=len(segments))
        store.save()
        segment_store = SegmentCheckpointStore.create(
            paths.artifact_path("tts_segments.v1.json"),
            stage=StageName.TTS.value,
            segment_ids=[str(segment["segment_id"]) for segment in segments],
        )
        if crash_stage == StageName.TTS.value and crash_after_segments == 0:
            segment_store.save()
            raise RuntimeError("simulated crash at tts after 0 segments")
        translations_by_id = {
            str(segment["segment_id"]): segment
            for segment in read_json(paths.artifact_path("translated.v1.json"))["segments"]
        }
        mix_audio_path = paths.tts_dir / "mix.wav"
        synthesize_tone_wav(mix_audio_path, duration_ms)
        tts_segments = []
        for segment in segments:
            segment_id = str(segment["segment_id"])
            orig_ms = int(segment["duration_ms"])
            raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
            aligned_audio_path = paths.tts_dir / f"{segment_id}.wav"
            provider_metadata = {}
            if self.provider_mode == "openai_compatible":
                if self.provider_bundle is None:
                    raise RuntimeError("provider bundle is not configured")
                tts_text = translations_by_id.get(segment_id, {}).get("vi_text", "")
                tts_result = asyncio.run(
                    self.provider_bundle.tts.synthesize(
                        tts_text,
                        voice="default",
                        output_path=raw_audio_path,
                    )
                )
                tts_duration_ms = tts_result.duration_ms or self.ffmpeg.duration_ms(tts_result.audio_path)
                provider_metadata = tts_result.provider_metadata
            else:
                tts_duration_ms = max(100, int(orig_ms * 1.1))
                synthesize_tone_wav(raw_audio_path, tts_duration_ms)
            timing_plan = plan_segment_duration(
                segment_id,
                orig_duration_ms=orig_ms,
                tts_duration_ms=tts_duration_ms,
            )
            if timing_plan.action == "time_stretch":
                apply_time_stretch(raw_audio_path, aligned_audio_path, timing_plan.stretch_ratio)
            else:
                apply_time_stretch(raw_audio_path, aligned_audio_path, 1.0)
            tts_segments.append(
                {
                    "segment_id": segment["segment_id"],
                    "target_start_ms": int(segment["start_ms"]) + 500,
                    "target_end_ms": int(segment["end_ms"]),
                    "original_start_ms": int(segment["start_ms"]),
                    "original_end_ms": int(segment["end_ms"]),
                    "commentary_delay_ms": 500,
                    "orig_duration_ms": orig_ms,
                    "tts_duration_ms": tts_duration_ms,
                    "alignment_action": timing_plan.action,
                    "stretch_ratio": timing_plan.stretch_ratio,
                    "overflow_ms": timing_plan.overflow_ms,
                    "raw_audio_path": paths.to_relative(raw_audio_path),
                    "audio_path": paths.to_relative(aligned_audio_path),
                    "warnings": timing_plan.warnings,
                    "provider_metadata": provider_metadata,
                }
            )
            segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=paths.to_relative(aligned_audio_path))
            segment_store.save()
        tts_manifest_path = paths.artifact_path("tts_manifest.v1.json")
        write_json_atomic(
            tts_manifest_path,
            {
                "schema_version": "1.0",
                "segments": tts_segments,
            },
        )
        manifest.record_artifact(
            name="tts_segments",
            version=1,
            path=segment_store.path,
            created_by_stage=StageName.TTS,
            schema_version="1.0",
        )
        manifest.record_artifact(
            name="tts_manifest",
            version=1,
            path=tts_manifest_path,
            created_by_stage=StageName.TTS,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(
            StageName.TTS,
            StageStatus.COMPLETED,
            artifact=paths.to_relative(tts_manifest_path),
            done=len(segments),
            total=len(segments),
        )
        store.save()
        return mix_audio_path

    def _stage_mixing(
        self,
        copied_input: Path,
        tts_audio: Path,
        duration_ms: int,
        paths: WorkspacePaths,
        store: CheckpointStore,
        manifest: ArtifactManifest,
    ) -> Path:
        store.mark_stage(StageName.MIXING, StageStatus.RUNNING)
        store.save()
        final_audio = paths.audio_dir / "final_mix.wav"
        self.ffmpeg.mix_commentary_audio(paths.audio_dir / "original.wav", tts_audio, final_audio)
        output_video = paths.output_dir / f"{copied_input.stem}_vi.mp4"
        self.ffmpeg.mux_video_audio(copied_input, final_audio, output_video)
        qa_path = paths.output_dir / f"{copied_input.stem}_vi.qa.json"
        write_json_atomic(
            qa_path,
            {
                "schema_version": "1.0",
                "job_id": paths.root.name,
                "input_duration_ms": duration_ms,
                "output_duration_ms": self.ffmpeg.duration_ms(output_video),
                "segments_total": len(read_json(paths.artifact_path("segments.v1.json"))["segments"]),
                "low_confidence_segments": 0,
                "glossary_terms": 1,
                "tts_overflow_segments": 0,
                "max_overflow_ms": 0,
                "sync_drift_p95_ms": 0,
                "warnings": [],
            },
        )
        manifest.record_artifact(
            name="final_audio",
            version=1,
            path=final_audio,
            created_by_stage=StageName.MIXING,
            schema_version="1.0",
        )
        manifest.record_artifact(
            name="qa_report",
            version=1,
            path=qa_path,
            created_by_stage=StageName.MIXING,
            schema_version="1.0",
        )
        manifest.record_artifact(
            name="output_video",
            version=1,
            path=output_video,
            created_by_stage=StageName.MIXING,
            schema_version="1.0",
        )
        manifest.save()
        store.mark_stage(StageName.MIXING, StageStatus.COMPLETED, artifact=paths.to_relative(output_video))
        store.save()
        return output_video
