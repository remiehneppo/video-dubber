from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from dubber.core.config import load_config
from dubber.core.concurrency import ProviderConcurrency
from dubber.core.enums import JobStatus, StageName, StageStatus
from dubber.core.io import read_json, write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.core.models import utc_now_iso
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.pipeline.resume_plan import ResumePlan
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
from dubber.tts.interactive_review import TTSReviewDecision, TTSReviewRequest
from dubber.tts.track_mixer import assemble_commentary_track


@dataclass(frozen=True)
class RunOptions:
    input_path: Path
    workspace_dir: Path
    config_path: Path | None = None
    domain: str | None = None
    domain_profile: str | None = None
    provider_mode: str = "mock"
    glossary_review: bool = False
    crash_stage: str | None = None
    crash_after_segments: int | None = None
    job_id: str | None = None


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
    def __init__(
        self,
        ffmpeg: FFmpegAdapter | None = None,
        provider_bundle: ProviderBundle | None = None,
        concurrency: ProviderConcurrency | None = None,
        tts_review_handler: Callable[[TTSReviewRequest], TTSReviewDecision] | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg or FFmpegAdapter()
        self.provider_bundle = provider_bundle
        self.provider_mode = "mock"
        self.concurrency = concurrency
        self.tts_review_handler = tts_review_handler

    def run(self, options: RunOptions, *, stop_after: StageName | None = None) -> RunSummary:
        config = load_config(options.config_path or Path("config.example.yaml"))
        if self.concurrency is None:
            self.concurrency = ProviderConcurrency(config.runtime)
        effective_domain = options.domain or config.project.domain
        effective_profile = options.domain_profile if options.domain_profile is not None else config.project.domain_profile
        config = replace(
            config,
            project=replace(config.project, domain=effective_domain, domain_profile=effective_profile),
        )
        if options.provider_mode not in {"mock", "openai_compatible"}:
            raise ValueError(f"Unsupported provider-mode: {options.provider_mode}")
        self.provider_mode = options.provider_mode
        if options.provider_mode == "openai_compatible" and self.provider_bundle is None:
            self.provider_bundle = build_provider_bundle(config)
        input_path = options.input_path.expanduser().resolve()
        self._validate_input(input_path, config.input.allowed_extensions, config.input.max_file_size_mb)

        job_id = options.job_id or f"job_{uuid.uuid4().hex[:12]}"
        paths = WorkspacePaths.create(options.workspace_dir, job_id)
        store = CheckpointStore.create(paths.job_state_file, job_id=job_id, input_file=Path("input") / input_path.name)
        manifest = ArtifactManifest.create(job_id, paths.manifest_file)
        copied_input = paths.input_dir / input_path.name
        shutil.copy2(input_path, copied_input)
        self._write_resolved_config(paths, options, config.project.domain, config.project.domain_profile)
        store.save()
        manifest.save()
        ctx = self._context(paths, store, manifest, config)
        return self._execute(
            ctx,
            start_stage=StageName.JOB_INIT,
            glossary_review=options.glossary_review,
            stop_after=stop_after,
            crash_stage=options.crash_stage,
            crash_after_segments=options.crash_after_segments,
        )

    def resume(
        self,
        workspace_dir: Path,
        job_id: str,
        *,
        from_stage: StageName | None = None,
        stop_after: StageName | None = None,
        no_cache: bool = False,
    ) -> RunSummary:
        paths = WorkspacePaths.create(workspace_dir, job_id)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        config = self._load_resolved_config(paths)
        if self.concurrency is None:
            self.concurrency = ProviderConcurrency(config.runtime)
        self._restore_provider_context(paths, config)
        ctx = self._context(paths, store, manifest, config)
        plan = self._build_resume_plan(
            ctx,
            from_stage=from_stage,
            no_cache=no_cache,
            invalidate_missing_artifacts=True,
        )
        if plan.already_completed or plan.start_stage is None:
            return self._summary(ctx, JobStatus.COMPLETED)
        start_stage = plan.start_stage
        if from_stage is not None:
            self._invalidate_from(
                ctx,
                start_stage,
                preserve_checkpoints=plan.preserve_checkpoints,
                no_cache=no_cache,
            )
        elif no_cache:
            self._invalidate_from(ctx, start_stage, preserve_checkpoints=False, no_cache=True)

        resolved = read_json(paths.root / "config.resolved.json") if (paths.root / "config.resolved.json").exists() else {}
        glossary_review = bool(resolved.get("glossary_review", False))
        if start_stage == StageName.GLOSSARY:
            locked = paths.artifact_path("glossary.locked.json")
            if locked.exists():
                self.publish_shared_glossary(workspace_dir, job_id, locked, locked=True)
                store = CheckpointStore.load(paths.job_state_file)
                manifest = ArtifactManifest.load(paths.manifest_file)
                ctx = self._context(paths, store, manifest, config)
                start_stage = StageName.TRANSLATION
            elif store.state.status == JobStatus.WAITING_REVIEW:
                return self._summary(ctx, JobStatus.WAITING_REVIEW)
        if start_stage == StageName.TRANSLATION and store.state.status == JobStatus.WAITING_REVIEW:
            if not paths.artifact_path("review.locked.json").exists():
                return self._summary(ctx, JobStatus.WAITING_REVIEW)
        return self._execute(
            ctx,
            start_stage=start_stage,
            glossary_review=glossary_review,
            stop_after=stop_after,
        )

    def plan_resume(
        self,
        workspace_dir: Path,
        job_id: str,
        *,
        from_stage: StageName | None = None,
        no_cache: bool = False,
    ) -> ResumePlan:
        paths = WorkspacePaths.create(workspace_dir, job_id, create_dirs=False)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        config = self._load_resolved_config(paths)
        self._restore_provider_context(paths, config)
        ctx = self._context(paths, store, manifest, config)
        return self._build_resume_plan(
            ctx,
            from_stage=from_stage,
            no_cache=no_cache,
            invalidate_missing_artifacts=False,
        )

    def invalidate(self, workspace_dir: Path, job_id: str, stage: StageName, *, no_cache: bool = False) -> None:
        paths = WorkspacePaths.create(workspace_dir, job_id)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        config = self._load_resolved_config(paths)
        self._restore_provider_context(paths, config)
        self._invalidate_from(self._context(paths, store, manifest, config), stage, no_cache=no_cache)

    def rerun_stage(self, workspace_dir: Path, job_id: str, stage: StageName) -> RunSummary:
        return self.resume(workspace_dir, job_id, from_stage=stage)

    def publish_shared_glossary(
        self,
        workspace_dir: Path,
        job_id: str,
        source: Path,
        *,
        locked: bool,
    ) -> None:
        paths = WorkspacePaths.create(workspace_dir, job_id)
        store = CheckpointStore.load(paths.job_state_file)
        manifest = ArtifactManifest.load(paths.manifest_file)
        filename = "glossary.locked.json" if locked else "glossary.draft.json"
        target = paths.artifact_path(filename)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        manifest.record_artifact(
            name="glossary" if locked else "glossary_draft",
            version=1,
            path=target,
            created_by_stage=StageName.GLOSSARY,
            schema_version="1.0",
        )
        store.mark_stage(
            StageName.GLOSSARY,
            StageStatus.COMPLETED if locked else StageStatus.WAITING_REVIEW,
            artifact=paths.to_relative(target),
        )
        store.mark_job(JobStatus.RUNNING if locked else JobStatus.WAITING_REVIEW)
        manifest.save()
        store.save()

    def _context(self, paths, store, manifest, config) -> StageContext:
        return StageContext(
            paths=paths,
            store=store,
            manifest=manifest,
            ffmpeg=self.ffmpeg,
            provider_mode=self.provider_mode,
            provider_bundle=self.provider_bundle,
            config=config,
            concurrency=self.concurrency,
        )

    def _execute(
        self,
        ctx: StageContext,
        *,
        start_stage: StageName,
        glossary_review: bool,
        stop_after: StageName | None = None,
        crash_stage: str | None = None,
        crash_after_segments: int | None = None,
    ) -> RunSummary:
        stages = list(StageName)
        start_index = stages.index(start_stage)
        copied_input = ctx.resolve_input()
        duration_ms = self.ffmpeg.duration_ms(copied_input)
        ctx.store.mark_job(JobStatus.RUNNING)
        ctx.store.save()
        try:
            if start_index <= stages.index(StageName.JOB_INIT):
                run_job_init(ctx, copied_input)
                if stop_after == StageName.JOB_INIT:
                    return self._summary(ctx, JobStatus.RUNNING)
            if start_index <= stages.index(StageName.AUDIO_EXTRACT):
                duration_ms = run_audio_extract(ctx, copied_input)
                if stop_after == StageName.AUDIO_EXTRACT:
                    return self._summary(ctx, JobStatus.RUNNING)
            if start_index <= stages.index(StageName.VAD):
                run_vad(ctx)
                if stop_after == StageName.VAD:
                    return self._summary(ctx, JobStatus.RUNNING)
            if start_index <= stages.index(StageName.ASR):
                run_asr(ctx)
                if stop_after == StageName.ASR:
                    return self._summary(ctx, JobStatus.RUNNING)
            if start_index <= stages.index(StageName.GLOSSARY):
                if run_glossary(ctx, glossary_review=glossary_review):
                    ctx.store.mark_job(JobStatus.WAITING_REVIEW)
                    ctx.store.save()
                    return self._summary(ctx, JobStatus.WAITING_REVIEW)
                if stop_after == StageName.GLOSSARY:
                    return self._summary(ctx, JobStatus.RUNNING)
            if start_index <= stages.index(StageName.TRANSLATION):
                if run_translation(ctx, total_duration_ms=duration_ms):
                    ctx.store.mark_job(JobStatus.WAITING_REVIEW)
                    ctx.store.save()
                    return self._summary(ctx, JobStatus.WAITING_REVIEW)
                if stop_after == StageName.TRANSLATION:
                    return self._summary(ctx, JobStatus.RUNNING)
            if start_index <= stages.index(StageName.TTS):
                tts_audio = run_tts(
                    ctx,
                    duration_ms,
                    crash_stage=crash_stage,
                    crash_after_segments=crash_after_segments,
                    tts_review_handler=self.tts_review_handler,
                )
                if stop_after == StageName.TTS:
                    return self._summary(ctx, JobStatus.RUNNING)
            else:
                tts_audio = ctx.paths.tts_dir / "mix.wav"
                if not tts_audio.exists():
                    tts_manifest = read_json(ctx.paths.artifact_path("tts_manifest.v1.json"))
                    assemble_commentary_track(
                        paths=ctx.paths,
                        ffmpeg=self.ffmpeg,
                        tts_segments=list(tts_manifest["segments"]),
                        output_audio=tts_audio,
                    )
            output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
        except Exception as exc:
            ctx.store.mark_job(JobStatus.FAILED, error=str(exc))
            ctx.store.save()
            raise
        ctx.store.mark_job(JobStatus.COMPLETED)
        ctx.store.save()
        ctx.manifest.save()
        return RunSummary(
            job_id=ctx.store.state.job_id,
            status=JobStatus.COMPLETED.value,
            output_video=ctx.paths.to_relative(output_video),
            workspace=str(ctx.paths.root),
        )

    def _first_invalid_stage(self, ctx: StageContext) -> StageName | None:
        return self._find_first_invalid_stage(ctx, invalidate_missing_artifacts=True)

    def _build_resume_plan(
        self,
        ctx: StageContext,
        *,
        from_stage: StageName | None,
        no_cache: bool,
        invalidate_missing_artifacts: bool,
    ) -> ResumePlan:
        if from_stage is not None:
            return ResumePlan(
                job_id=ctx.store.state.job_id,
                start_stage=from_stage,
                from_stage=from_stage,
                no_cache=no_cache,
                preserve_checkpoints=from_stage == StageName.TTS and not no_cache,
                already_completed=False,
                reason="from_stage_requested",
            )
        start_stage = self._find_first_invalid_stage(
            ctx,
            invalidate_missing_artifacts=invalidate_missing_artifacts,
        )
        if start_stage is None:
            if not no_cache:
                return ResumePlan(
                    job_id=ctx.store.state.job_id,
                    start_stage=None,
                    from_stage=None,
                    no_cache=False,
                    preserve_checkpoints=False,
                    already_completed=True,
                    reason="all_stages_valid",
                )
            return ResumePlan(
                job_id=ctx.store.state.job_id,
                start_stage=StageName.TTS,
                from_stage=None,
                no_cache=True,
                preserve_checkpoints=False,
                already_completed=False,
                reason="completed_job_no_cache_defaults_to_tts",
            )
        return ResumePlan(
            job_id=ctx.store.state.job_id,
            start_stage=start_stage,
            from_stage=None,
            no_cache=no_cache,
            preserve_checkpoints=not no_cache,
            already_completed=False,
            reason="first_invalid_stage",
        )

    def _find_first_invalid_stage(
        self,
        ctx: StageContext,
        *,
        invalidate_missing_artifacts: bool,
    ) -> StageName | None:
        required = {
            StageName.JOB_INIT: (("input_metadata", 1),),
            StageName.AUDIO_EXTRACT: (("audio_analysis", 1),),
            StageName.VAD: (("segments", 1),),
            StageName.ASR: (("asr_segments", 1), ("transcript", 1), ("source_normalization", 1)),
            StageName.GLOSSARY: (("glossary", 1),),
            StageName.TRANSLATION: (
                ("translation_blocks", 1),
                ("dubbing_cues", 2),
                ("speech_timeline", 1),
                ("translated", 2),
            ),
            StageName.TTS: (("tts_segments", 1), ("tts_manifest", 1), ("spoken_cues", 1)),
            StageName.MIXING: (("output_video", 1),),
        }
        for stage in StageName:
            progress = ctx.store.state.stages[stage]
            if stage in {StageName.GLOSSARY, StageName.TRANSLATION} and progress.status == StageStatus.WAITING_REVIEW:
                return stage
            if progress.status != StageStatus.COMPLETED:
                return stage
            if any(not ctx.manifest.validate_artifact(name, version) for name, version in required[stage]):
                if invalidate_missing_artifacts:
                    self._invalidate_from(ctx, stage, preserve_checkpoints=True)
                return stage
            if stage == StageName.AUDIO_EXTRACT and not all(
                (ctx.paths.audio_dir / name).exists() for name in ("original.wav", "vocals.wav")
            ):
                if invalidate_missing_artifacts:
                    self._invalidate_from(ctx, stage, preserve_checkpoints=True)
                return stage
            if stage == StageName.TTS and not (ctx.paths.tts_dir / "mix.wav").exists():
                if invalidate_missing_artifacts:
                    self._invalidate_from(ctx, stage, preserve_checkpoints=True)
                return stage
        return None

    def _invalidate_from(
        self,
        ctx: StageContext,
        stage: StageName,
        *,
        preserve_checkpoints: bool = False,
        no_cache: bool = False,
    ) -> None:
        checkpoint_names = {
            StageName.ASR: "asr_segments",
            StageName.GLOSSARY: "glossary_blocks",
            StageName.TRANSLATION: "translation_blocks",
            StageName.TTS: "tts_segments",
        }
        preserve: set[str] = set()
        checkpoint_name = checkpoint_names.get(stage)
        if preserve_checkpoints and checkpoint_name and ctx.manifest.validate_artifact(checkpoint_name, 1):
            preserve.add(checkpoint_name)
        ctx.manifest.remove_from_stage(stage, preserve_names=preserve)
        if no_cache:
            self._remove_stage_caches(ctx, stage)
        ctx.manifest.save()
        ctx.store.reset_from(stage)
        ctx.store.save()

    def _remove_stage_caches(self, ctx: StageContext, stage: StageName) -> None:
        affected = set(list(StageName)[list(StageName).index(stage):])
        checkpoint_files = {
            StageName.ASR: "asr_segments.v1.json",
            StageName.GLOSSARY: "glossary_blocks.v1.json",
            StageName.TRANSLATION: "translation_blocks.v1.json",
            StageName.TTS: "tts_segments.v1.json",
        }
        for checkpoint_stage, filename in checkpoint_files.items():
            if checkpoint_stage not in affected:
                continue
            path = ctx.paths.artifact_path(filename)
            if path.exists():
                path.unlink()
        if StageName.TTS in affected:
            for filename in (
                "tts_interactive_overrides.v1.json",
                "tts_manifest.v1.json",
                "spoken_cues.v1.json",
            ):
                path = ctx.paths.artifact_path(filename)
                if path.exists():
                    path.unlink()
            for directory in (ctx.paths.tts_dir, ctx.paths.raw_dir / "tts"):
                if directory.exists():
                    shutil.rmtree(directory)
            ctx.paths.tts_dir.mkdir(parents=True, exist_ok=True)
            ctx.paths.raw_dir.mkdir(parents=True, exist_ok=True)


    def _summary(self, ctx: StageContext, status: JobStatus) -> RunSummary:
        output = ""
        artifact = ctx.manifest.get("output_video", 1)
        if artifact is not None and ctx.manifest.validate_artifact("output_video", 1):
            output = artifact.path
        return RunSummary(
            job_id=ctx.store.state.job_id,
            status=status.value,
            output_video=output,
            workspace=str(ctx.paths.root),
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
        cues = read_json(paths.artifact_path("dubbing_cues.v2.json"))["cues"]
        matching = [cue for cue in cues if str(cue["cue_id"]) == segment_id]
        if not matching:
            raise ValueError(f"Unknown segment_id: {segment_id}")

        checkpoint = SegmentCheckpointStore.load(paths.artifact_path("tts_segments.v1.json"))
        checkpoint.mark(segment_id, StageStatus.PENDING)
        checkpoint.save()
        tts_audio = run_tts(ctx, duration_ms)
        output_video = run_mixing(ctx, copied_input, tts_audio, duration_ms)
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

    def _write_resolved_config(
        self,
        paths: WorkspacePaths,
        options: RunOptions,
        default_domain: str,
        default_domain_profile: str,
    ) -> None:
        config_path = options.config_path.expanduser().resolve() if options.config_path is not None else None
        write_json_atomic(
            paths.root / "config.resolved.json",
            {
                "provider_mode": options.provider_mode,
                "domain": options.domain or default_domain,
                "domain_profile": options.domain_profile if options.domain_profile is not None else default_domain_profile,
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
        domain_profile = str(resolved.get("domain_profile", config.project.domain_profile))
        return replace(config, project=replace(config.project, domain=domain, domain_profile=domain_profile))

    def _restore_provider_context(self, paths: WorkspacePaths, config) -> None:
        resolved_path = paths.root / "config.resolved.json"
        if resolved_path.exists():
            resolved = read_json(resolved_path)
            self.provider_mode = str(resolved.get("provider_mode", self.provider_mode))
        if self.provider_mode == "openai_compatible" and self.provider_bundle is None:
            self.provider_bundle = build_provider_bundle(config)


@dataclass(frozen=True)
class BatchOptions:
    input_dir: Path
    workspace_dir: Path
    config_path: Path | None = None
    domain: str | None = None
    domain_profile: str | None = None
    provider_mode: str = "mock"
    glossary_review: bool = False


@dataclass(frozen=True)
class BatchSummary:
    batch_id: str
    status: str
    counts: dict[str, int]
    workspace: str
    jobs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "counts": self.counts,
            "workspace": self.workspace,
            "jobs": self.jobs,
        }


class BatchManager:
    def scan_inputs(self, input_dir: Path, allowed_extensions: list[str]) -> list[Path]:
        directory = input_dir.expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {directory}")
        allowed = {extension.lower() for extension in allowed_extensions}
        return sorted(
            [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed],
            key=lambda path: (path.name.casefold(), path.name),
        )

    def run(self, options: BatchOptions) -> BatchSummary:
        config_path = options.config_path or Path("config.example.yaml")
        config = load_config(config_path)
        concurrency = ProviderConcurrency(config.runtime)
        inputs = self.scan_inputs(options.input_dir, config.input.allowed_extensions)
        if not inputs:
            raise ValueError("No supported video files found in input directory")
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        root = options.workspace_dir / batch_id
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        jobs = [
            {
                "input_file": str(path),
                "input_name": path.name,
                "job_id": f"job_{uuid.uuid4().hex[:12]}",
                "status": "pending",
                "error": None,
                "output_video": "",
            }
            for path in inputs
        ]
        now = utc_now_iso()
        state: dict[str, Any] = {
            "schema_version": "1.0",
            "batch_id": batch_id,
            "status": "running",
            "input_dir": str(options.input_dir.expanduser().resolve()),
            "provider_mode": options.provider_mode,
            "glossary_review": options.glossary_review,
            "config_path": str(options.config_path.expanduser().resolve()) if options.config_path else None,
            "domain": options.domain or config.project.domain,
            "domain_profile": options.domain_profile if options.domain_profile is not None else config.project.domain_profile,
            "created_at": now,
            "updated_at": now,
            "jobs": jobs,
        }
        write_json_atomic(root / "batch_state.json", state)
        write_json_atomic(
            root / "batch_manifest.json",
            {
                "schema_version": "1.0",
                "batch_id": batch_id,
                "inputs": [
                    {"input_file": job["input_file"], "input_name": job["input_name"], "job_id": job["job_id"]}
                    for job in jobs
                ],
            },
        )

        def prepare(job: dict[str, Any]) -> RunSummary:
            return JobManager(concurrency=concurrency).run(
                RunOptions(
                    input_path=Path(job["input_file"]),
                    workspace_dir=root / "jobs",
                    config_path=options.config_path,
                    domain=options.domain,
                    domain_profile=options.domain_profile,
                    provider_mode=options.provider_mode,
                    glossary_review=False,
                    job_id=str(job["job_id"]),
                ),
                stop_after=StageName.ASR,
            )

        self._run_jobs(state, root, jobs, prepare, config.runtime.max_parallel_jobs, success_status="asr_completed")
        successful = [job for job in jobs if job["status"] == "asr_completed"]
        if not successful:
            state["status"] = "failed"
            self._save_state(root, state)
            return self._summary(root, state)

        glossary = self._build_shared_glossary(root, state, config, successful, concurrency)
        state["glossary"] = f"artifacts/{glossary.name}"
        if options.glossary_review:
            for job in successful:
                JobManager(concurrency=concurrency).publish_shared_glossary(
                    root / "jobs", str(job["job_id"]), glossary, locked=False
                )
                job["status"] = JobStatus.WAITING_REVIEW.value
            state["status"] = JobStatus.WAITING_REVIEW.value
            self._save_state(root, state)
            return self._summary(root, state)

        self._complete_jobs(
            root, state, successful, glossary, config.runtime.max_parallel_jobs, concurrency=concurrency
        )
        self._set_final_status(state)
        self._save_state(root, state)
        return self._summary(root, state)

    def resume(
        self,
        workspace_dir: Path,
        batch_id: str,
        *,
        job_ids: list[str] | None = None,
        from_stage: StageName | None = None,
        no_cache: bool = False,
    ) -> BatchSummary:
        root, state = self._load(workspace_dir, batch_id)
        config = self._batch_config(state)
        concurrency = ProviderConcurrency(config.runtime)
        jobs = list(state["jobs"])
        selected = [
            job for job in jobs
            if (job_ids is None and (from_stage is not None or job["status"] != JobStatus.COMPLETED.value))
            or (job_ids is not None and job["job_id"] in job_ids)
        ]
        if job_ids is not None:
            missing = sorted(set(job_ids) - {str(job["job_id"]) for job in jobs})
            if missing:
                raise ValueError(f"Unknown batch job ids: {', '.join(missing)}")
        locked = root / "artifacts" / "glossary.locked.json"
        if not locked.exists():
            configured = state.get("glossary")
            if configured and str(configured).endswith("glossary.locked.json"):
                locked = root / str(configured)
        if not locked.exists():
            draft = root / "artifacts" / "glossary.draft.json"
            if bool(state.get("glossary_review")) and draft.exists():
                raise FileNotFoundError("Batch locked glossary is missing; review artifacts/glossary.draft.json and save glossary.locked.json")
            glossary_jobs = [
                job for job in jobs
                if str(job.get("status")) in {
                    "asr_completed",
                    "ready",
                    JobStatus.WAITING_REVIEW.value,
                    JobStatus.COMPLETED.value,
                }
                and (root / "jobs" / str(job["job_id"]) / "artifacts" / "transcript.v1.json").exists()
            ]
            if not glossary_jobs:
                raise FileNotFoundError(
                    "Batch locked glossary is missing and no ASR-completed jobs are available to rebuild it"
                )
            rebuilt = self._build_shared_glossary(root, state, config, glossary_jobs, concurrency)
            state["glossary"] = f"artifacts/{rebuilt.name}"
            if bool(state.get("glossary_review")):
                for job in glossary_jobs:
                    JobManager(concurrency=concurrency).publish_shared_glossary(
                        root / "jobs", str(job["job_id"]), rebuilt, locked=False
                    )
                    job["status"] = JobStatus.WAITING_REVIEW.value
                state["status"] = JobStatus.WAITING_REVIEW.value
                self._save_state(root, state)
                return self._summary(root, state)
            locked = rebuilt

        ready: list[dict[str, Any]] = []
        for job in selected:
            manager = JobManager(concurrency=concurrency)
            job_id = str(job["job_id"])
            try:
                job_state = CheckpointStore.load(root / "jobs" / job_id / "job_state.json").state
                needs_asr = (
                    from_stage is not None and list(StageName).index(from_stage) <= list(StageName).index(StageName.ASR)
                ) or (
                    from_stage is None
                    and job_state.status != JobStatus.COMPLETED
                    and list(StageName).index(job_state.current_stage) <= list(StageName).index(StageName.ASR)
                )
                if needs_asr:
                    manager.resume(root / "jobs", job_id, from_stage=from_stage, stop_after=StageName.ASR, no_cache=no_cache)
                elif from_stage == StageName.GLOSSARY:
                    manager.invalidate(root / "jobs", job_id, StageName.GLOSSARY, no_cache=no_cache)
                manager.publish_shared_glossary(root / "jobs", job_id, locked, locked=True)
                ready.append(job)
                job["status"] = "ready"
                job["error"] = None
            except Exception as exc:
                job["status"] = JobStatus.FAILED.value
                job["error"] = str(exc)
                self._save_state(root, state)
        self._complete_jobs(
            root,
            state,
            ready,
            locked,
            self._batch_config(state).runtime.max_parallel_jobs,
            concurrency=concurrency,
            from_stage=from_stage if from_stage not in {StageName.JOB_INIT, StageName.AUDIO_EXTRACT, StageName.VAD, StageName.ASR, StageName.GLOSSARY} else None,
            no_cache=no_cache,
            publish_glossary=False,
        )
        state["glossary"] = "artifacts/glossary.locked.json"
        self._set_final_status(state)
        self._save_state(root, state)
        return self._summary(root, state)

    def plan_resume(
        self,
        workspace_dir: Path,
        batch_id: str,
        *,
        job_ids: list[str] | None = None,
        from_stage: StageName | None = None,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        root, state = self._load(workspace_dir, batch_id)
        config = self._batch_config(state)
        concurrency = ProviderConcurrency(config.runtime)
        jobs = list(state["jobs"])
        selected = [
            job for job in jobs
            if (job_ids is None and (from_stage is not None or job["status"] != JobStatus.COMPLETED.value))
            or (job_ids is not None and job["job_id"] in job_ids)
        ]
        if job_ids is not None:
            missing = sorted(set(job_ids) - {str(job["job_id"]) for job in jobs})
            if missing:
                raise ValueError(f"Unknown batch job ids: {', '.join(missing)}")
        plans: list[dict[str, object]] = []
        for job in selected:
            job_id = str(job["job_id"])
            item: dict[str, object] = {
                "job_id": job_id,
                "input_name": str(job.get("input_name") or job.get("input_file") or ""),
                "status": str(job.get("status") or ""),
            }
            try:
                item["plan"] = JobManager(concurrency=concurrency).plan_resume(
                    root / "jobs",
                    job_id,
                    from_stage=from_stage,
                    no_cache=no_cache,
                ).to_dict()
            except Exception as exc:
                item["error"] = str(exc)
            plans.append(item)
        return {
            "batch_id": batch_id,
            "workspace": str(root),
            "selected_jobs": len(plans),
            "jobs": plans,
        }

    def status(self, workspace_dir: Path, batch_id: str) -> BatchSummary:
        root, state = self._load(workspace_dir, batch_id)
        return self._summary(root, state)

    def validate(self, workspace_dir: Path, batch_id: str) -> dict[str, Any]:
        root, state = self._load(workspace_dir, batch_id)
        errors: list[str] = []
        for job in state["jobs"]:
            manifest_path = root / "jobs" / str(job["job_id"]) / "manifest.json"
            if not manifest_path.exists():
                errors.append(f"{job['job_id']}: manifest missing")
                continue
            manifest = ArtifactManifest.load(manifest_path)
            for artifact in manifest.artifacts:
                if not manifest.validate_artifact(artifact.name, artifact.version):
                    errors.append(f"{job['job_id']}: {artifact.name}.v{artifact.version}")
        glossary = state.get("glossary")
        if glossary and not (root / str(glossary)).exists():
            errors.append(f"batch: missing {glossary}")
        return {"batch_id": batch_id, "valid": not errors, "errors": errors}

    def _run_jobs(
        self,
        state: dict[str, Any],
        root: Path,
        jobs: list[dict[str, Any]],
        operation,
        max_workers: int,
        *,
        success_status: str,
    ) -> None:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(operation, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    summary = future.result()
                    job["status"] = (
                        JobStatus.WAITING_REVIEW.value
                        if summary.status == JobStatus.WAITING_REVIEW.value
                        else success_status
                    )
                    job["output_video"] = summary.output_video
                    job["error"] = None
                except Exception as exc:
                    job["status"] = JobStatus.FAILED.value
                    job["error"] = str(exc)
                self._save_state(root, state)

    def _complete_jobs(
        self,
        root: Path,
        state: dict[str, Any],
        jobs: list[dict[str, Any]],
        glossary: Path,
        max_workers: int,
        *,
        concurrency: ProviderConcurrency,
        from_stage: StageName | None = None,
        no_cache: bool = False,
        publish_glossary: bool = True,
    ) -> None:
        def complete(job: dict[str, Any]) -> RunSummary:
            manager = JobManager(concurrency=concurrency)
            if publish_glossary:
                manager.publish_shared_glossary(root / "jobs", str(job["job_id"]), glossary, locked=True)
            return manager.resume(root / "jobs", str(job["job_id"]), from_stage=from_stage, no_cache=no_cache)

        self._run_jobs(state, root, jobs, complete, max_workers, success_status=JobStatus.COMPLETED.value)
        for job in jobs:
            if job["status"] == JobStatus.COMPLETED.value:
                job_state = CheckpointStore.load(root / "jobs" / str(job["job_id"]) / "job_state.json").state
                manifest = ArtifactManifest.load(root / "jobs" / str(job["job_id"]) / "manifest.json")
                output = manifest.get("output_video", 1)
                job["output_video"] = output.path if output is not None else ""

    def _build_shared_glossary(
        self,
        root: Path,
        state: dict[str, Any],
        config,
        jobs: list[dict[str, Any]],
        concurrency: ProviderConcurrency,
    ) -> Path:
        shared_paths = WorkspacePaths.create(root, ".shared_glossary")
        combined: list[dict[str, Any]] = []
        for job in jobs:
            transcript = read_json(root / "jobs" / str(job["job_id"]) / "artifacts" / "transcript.v1.json")
            for segment in transcript["segments"]:
                item = dict(segment)
                item["segment_id"] = f"{job['job_id']}::{segment['segment_id']}"
                combined.append(item)
        write_json_atomic(shared_paths.artifact_path("transcript.v1.json"), {"schema_version": "1.0", "segments": combined})
        if shared_paths.job_state_file.exists():
            store = CheckpointStore.load(shared_paths.job_state_file)
        else:
            store = CheckpointStore.create(
                shared_paths.job_state_file,
                job_id=f"{state['batch_id']}_glossary",
                input_file=Path("input/none"),
            )
            store.save()
        if shared_paths.manifest_file.exists():
            manifest = ArtifactManifest.load(shared_paths.manifest_file)
        else:
            manifest = ArtifactManifest.create(f"{state['batch_id']}_glossary", shared_paths.manifest_file)
            manifest.save()
        provider_bundle = build_provider_bundle(config) if state["provider_mode"] == "openai_compatible" else None
        ctx = StageContext(
            paths=shared_paths,
            store=store,
            manifest=manifest,
            ffmpeg=None,
            provider_mode=str(state["provider_mode"]),
            provider_bundle=provider_bundle,
            config=replace(
                config,
                project=replace(
                    config.project,
                    domain=str(state["domain"]),
                    domain_profile=str(state.get("domain_profile", config.project.domain_profile)),
                ),
            ),
            concurrency=concurrency,
        )
        review = bool(state["glossary_review"])
        run_glossary(ctx, glossary_review=review)
        source = shared_paths.artifact_path("glossary.draft.json" if review else "glossary.locked.json")
        target = root / "artifacts" / source.name
        shutil.copy2(source, target)
        return target

    def _batch_config(self, state: dict[str, Any]):
        value = state.get("config_path")
        config = load_config(Path(str(value)) if value else Path("config.example.yaml"))
        return replace(
            config,
            project=replace(
                config.project,
                domain=str(state.get("domain", config.project.domain)),
                domain_profile=str(state.get("domain_profile", config.project.domain_profile)),
            ),
        )

    def _set_final_status(self, state: dict[str, Any]) -> None:
        statuses = [str(job["status"]) for job in state["jobs"]]
        completed = statuses.count(JobStatus.COMPLETED.value)
        failed = statuses.count(JobStatus.FAILED.value)
        if completed == len(statuses):
            state["status"] = JobStatus.COMPLETED.value
        elif failed == len(statuses):
            state["status"] = JobStatus.FAILED.value
        elif failed:
            state["status"] = JobStatus.PARTIAL_FAILED.value
        elif JobStatus.WAITING_REVIEW.value in statuses:
            state["status"] = JobStatus.WAITING_REVIEW.value
        else:
            state["status"] = "running"

    def _load(self, workspace_dir: Path, batch_id: str) -> tuple[Path, dict[str, Any]]:
        if "/" in batch_id or ".." in batch_id or not batch_id.startswith("batch_"):
            raise ValueError("Invalid batch id")
        root = workspace_dir / batch_id
        path = root / "batch_state.json"
        if not path.exists():
            raise FileNotFoundError(f"Batch state not found: {batch_id}")
        return root, read_json(path)

    def _save_state(self, root: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now_iso()
        write_json_atomic(root / "batch_state.json", state)

    def _summary(self, root: Path, state: dict[str, Any]) -> BatchSummary:
        counts: dict[str, int] = {}
        for job in state["jobs"]:
            status = str(job["status"])
            counts[status] = counts.get(status, 0) + 1
        return BatchSummary(
            batch_id=str(state["batch_id"]),
            status=str(state["status"]),
            counts=counts,
            workspace=str(root),
            jobs=list(state["jobs"]),
        )
