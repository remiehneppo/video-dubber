from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from dubber.core.config import load_config
from dubber.core.enums import JobStatus, StageName, StageStatus
from dubber.core.io import read_json, write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import (
    run_asr,
    run_audio_extract,
    run_glossary,
    run_job_init,
    run_mixing,
    run_translation,
    run_tts,
    run_vad,
)
from dubber.providers.factory import ProviderBundle, build_provider_bundle
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.segment_producer import produce_mock_tts_segment
from dubber.tts.track_mixer import assemble_commentary_track

logger = logging.getLogger(__name__)


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
        effective_domain = options.domain or config.project.domain
        config = replace(config, project=replace(config.project, domain=effective_domain))
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
            config=config,
        )

        copied_input = paths.input_dir / input_path.name
        shutil.copy2(input_path, copied_input)
        self._write_resolved_config(paths, options, config.project.domain)
        logger.info(
            "job %s started input=%s workspace=%s provider_mode=%s request_timeout_sec=%s retry_max_attempts=%s",
            job_id,
            input_path,
            paths.root,
            self.provider_mode,
            config.runtime.request_timeout_sec,
            config.runtime.retry_max_attempts,
        )

        try:
            run_job_init(ctx, copied_input)
            duration_ms = run_audio_extract(ctx, copied_input)
            run_vad(ctx)
            run_asr(ctx)
            if run_glossary(ctx, glossary_review=options.glossary_review):
                store.mark_job(JobStatus.WAITING_REVIEW)
                store.save()
                manifest.save()
                logger.info("job %s waiting for glossary review workspace=%s", job_id, paths.root)
                return RunSummary(
                    job_id=job_id,
                    status=JobStatus.WAITING_REVIEW.value,
                    output_video="",
                    workspace=str(paths.root),
                )
            run_translation(ctx)
            tts_audio = run_tts(
                ctx,
                duration_ms,
                crash_stage=options.crash_stage,
                crash_after_segments=options.crash_after_segments,
            )
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
        except Exception as exc:
            logger.exception("job %s failed at stage=%s", job_id, store.state.current_stage.value)
            store.mark_job(JobStatus.FAILED, error=str(exc))
            store.save()
            raise

        store.mark_job(JobStatus.COMPLETED)
        store.save()
        manifest.save()
        logger.info("job %s completed output=%s", job_id, output_video)
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
        config = self._load_resolved_config(paths)
        self._restore_provider_context(paths, config)
        ctx = StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
            config=config,
        )
        if store.state.status in {JobStatus.FAILED, JobStatus.RUNNING} and store.state.current_stage == StageName.TTS:
            copied_input = paths.resolve_relative(store.state.input_file)
            duration_ms = self.ffmpeg.duration_ms(copied_input)
            store.mark_job(JobStatus.RUNNING)
            store.save()
            tts_audio = run_tts(ctx, duration_ms)
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
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
        tts_audio = run_tts(ctx, duration_ms)
        output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
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
        config = self._load_resolved_config(paths)
        self._restore_provider_context(paths, config)
        ctx = StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
            config=config,
        )
        copied_input = paths.resolve_relative(store.state.input_file)
        duration_ms = self.ffmpeg.duration_ms(copied_input)

        store.mark_job(JobStatus.RUNNING)
        store.save()
        if stage == StageName.TRANSLATION:
            run_translation(ctx)
            tts_audio = run_tts(ctx, duration_ms)
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
        elif stage == StageName.TTS:
            tts_audio = run_tts(ctx, duration_ms)
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
        elif stage == StageName.MIXING:
            tts_manifest = read_json(paths.artifact_path("tts_manifest.v1.json"))
            tts_audio = paths.tts_dir / "mix.wav"
            assemble_commentary_track(
                paths=paths,
                ffmpeg=self.ffmpeg,
                tts_segments=list(tts_manifest["segments"]),
                output_audio=tts_audio,
            )
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
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
        config = self._load_resolved_config(paths)
        self._restore_provider_context(paths, config)
        ctx = StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
            config=config,
        )
        copied_input = paths.resolve_relative(store.state.input_file)
        duration_ms = self.ffmpeg.duration_ms(copied_input)
        segments = read_json(paths.artifact_path("segments.v1.json"))["segments"]
        matching = [segment for segment in segments if str(segment["segment_id"]) == segment_id]
        if not matching:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        segment = matching[0]

        checkpoint = SegmentCheckpointStore.load(paths.artifact_path("tts_segments.v1.json"))
        if ctx.provider_mode == "openai_compatible":
            transcript_by_id = {
                str(item["segment_id"]): item
                for item in read_json(paths.artifact_path("transcript.v1.json"))["segments"]
            }
            translations_by_id = {
                str(item["segment_id"]): item
                for item in read_json(paths.artifact_path("translated.v1.json"))["segments"]
            }
            source_text = str(transcript_by_id.get(segment_id, {}).get("source_text", ""))
            translated_text = str(translations_by_id.get(segment_id, {}).get("vi_text", ""))
            row = asyncio.run(
                produce_provider_tts_segment(
                    paths=paths,
                    segment=segment,
                    text="" if not source_text.strip() else translated_text,
                    provider_bundle=ctx.require_provider_bundle(),
                    ffmpeg=self.ffmpeg,
                )
            )
        else:
            row = produce_mock_tts_segment(paths=paths, segment=segment)
        checkpoint.mark(segment_id, StageStatus.COMPLETED, artifact=row["audio_path"])
        checkpoint.save()

        tts_manifest_path = paths.artifact_path("tts_manifest.v1.json")
        tts_manifest = read_json(tts_manifest_path)
        for item in tts_manifest["segments"]:
            if str(item["segment_id"]) == segment_id:
                item.update(row)
        write_json_atomic(tts_manifest_path, tts_manifest)
        tts_audio = paths.tts_dir / "mix.wav"
        assemble_commentary_track(
            paths=paths,
            ffmpeg=self.ffmpeg,
            tts_segments=list(tts_manifest["segments"]),
            output_audio=tts_audio,
        )
        manifest.record_artifact(name="tts_segments", version=1, path=checkpoint.path, created_by_stage=StageName.TTS, schema_version="1.0")
        manifest.record_artifact(name="tts_manifest", version=1, path=tts_manifest_path, created_by_stage=StageName.TTS, schema_version="1.0")
        output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
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
        config_path = options.config_path.expanduser().resolve() if options.config_path is not None else None
        write_json_atomic(
            paths.root / "config.resolved.json",
            {
                "provider_mode": options.provider_mode,
                "domain": options.domain or default_domain,
                "glossary_review": options.glossary_review,
                "config_path": str(config_path) if config_path is not None else None,
            },
        )

    def _load_resolved_config(self, paths: WorkspacePaths):
        resolved_path = paths.root / "config.resolved.json"
        if not resolved_path.exists():
            return load_config(Path("config.example.yaml"))
        resolved = read_json(resolved_path)
        config_path_value = resolved.get("config_path")
        config_path = Path(str(config_path_value)) if config_path_value else Path("config.example.yaml")
        config = load_config(config_path)
        domain = str(resolved.get("domain", config.project.domain))
        return replace(config, project=replace(config.project, domain=domain))

    def _restore_provider_context(self, paths: WorkspacePaths, config) -> None:
        resolved_path = paths.root / "config.resolved.json"
        if resolved_path.exists():
            resolved = read_json(resolved_path)
            self.provider_mode = str(resolved.get("provider_mode", self.provider_mode))
        if self.provider_mode == "openai_compatible" and self.provider_bundle is None:
            self.provider_bundle = build_provider_bundle(config)
