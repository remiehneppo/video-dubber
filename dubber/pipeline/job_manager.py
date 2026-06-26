from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
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
            config=config,
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
            tts_audio = run_tts(
                ctx,
                duration_ms,
                crash_stage=options.crash_stage,
                crash_after_segments=options.crash_after_segments,
            )
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
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
            tts_audio = run_tts(ctx, duration_ms)
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
        elif stage == StageName.TTS:
            tts_audio = run_tts(ctx, duration_ms)
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
        elif stage == StageName.MIXING:
            tts_manifest = read_json(paths.artifact_path("tts_manifest.v1.json"))
            first_audio = paths.resolve_relative(tts_manifest["segments"][0]["audio_path"])
            output_video = run_mixing(ctx, copied_input, first_audio, duration_ms)
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
        segments = read_json(paths.artifact_path("segments.v1.json"))["segments"]
        matching = [segment for segment in segments if str(segment["segment_id"]) == segment_id]
        if not matching:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        segment = matching[0]

        checkpoint = SegmentCheckpointStore.load(paths.artifact_path("tts_segments.v1.json"))
        row = produce_mock_tts_segment(paths=paths, segment=segment)
        audio_path = paths.resolve_relative(row["audio_path"])
        checkpoint.mark(segment_id, StageStatus.COMPLETED, artifact=row["audio_path"])
        checkpoint.save()

        tts_manifest_path = paths.artifact_path("tts_manifest.v1.json")
        tts_manifest = read_json(tts_manifest_path)
        for item in tts_manifest["segments"]:
            if str(item["segment_id"]) == segment_id:
                item.update(row)
        write_json_atomic(tts_manifest_path, tts_manifest)
        manifest.record_artifact(name="tts_segments", version=1, path=checkpoint.path, created_by_stage=StageName.TTS, schema_version="1.0")
        manifest.record_artifact(name="tts_manifest", version=1, path=tts_manifest_path, created_by_stage=StageName.TTS, schema_version="1.0")
        output_video = run_mixing(ctx, copied_input, audio_path, duration_ms)
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
