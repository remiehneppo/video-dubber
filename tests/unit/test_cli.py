from __future__ import annotations

import io
import sys
import json
from pathlib import Path

import pytest

import cli
from cli import main
from dubber.core.enums import JobStatus, StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.pipeline.job_manager import BatchManager, JobManager


def test_jobs_lists_existing_job_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_one")
    CheckpointStore.create(paths.job_state_file, job_id="job_one", input_file=Path("input/a.mp4")).save()

    exit_code = main(["jobs", "--workspace", str(workspace)])

    assert exit_code == 0
    assert "job_one" in capsys.readouterr().out


def test_status_prints_job_state_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_status")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_status", input_file=Path("input/a.mp4"))
    store.mark_stage(StageName.JOB_INIT, StageStatus.COMPLETED, artifact="input_metadata.v1.json")
    store.save()

    exit_code = main(["status", "--workspace", str(workspace), "--job", "job_status"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job_status"
    assert payload["stages"]["job_init"]["status"] == "completed"


def test_workspace_status_reports_single_job_with_video_and_resume_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_failed")
    store = CheckpointStore.create(
        paths.job_state_file, job_id="job_failed", input_file=Path("input/calculus.mp4")
    )
    store.mark_stage(StageName.ASR, StageStatus.COMPLETED, artifact="transcript.v1.json", done=3, total=3)
    store.mark_stage(StageName.TTS, StageStatus.FAILED, done=4, total=5, error="tts_semantic_quality_failed")
    store.mark_job(JobStatus.FAILED, error="cue_123: tts_semantic_quality_failed")
    store.save()

    exit_code = main(["workspace-status", str(workspace)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "job_failed" in out
    assert "calculus.mp4" in out
    assert "failed" in out
    assert "tts 4/5" in out
    assert "cue_123: tts_semantic_quality_failed" in out
    assert f"python cli.py resume --workspace {workspace} --job job_failed" in out


def test_workspace_status_reports_batch_jobs_with_video_names_and_batch_resume_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    batch_root = workspace / "batch_demo"
    job_root = batch_root / "jobs"
    first_paths = WorkspacePaths.create(job_root, "job_one")
    first_store = CheckpointStore.create(
        first_paths.job_state_file, job_id="job_one", input_file=Path("input/lesson-one.mp4")
    )
    first_store.mark_job(JobStatus.COMPLETED)
    first_store.save()
    second_paths = WorkspacePaths.create(job_root, "job_two")
    second_store = CheckpointStore.create(
        second_paths.job_state_file, job_id="job_two", input_file=Path("input/lesson-two.mp4")
    )
    second_store.mark_stage(StageName.TRANSLATION, StageStatus.WAITING_REVIEW, done=1, total=2)
    second_store.mark_job(JobStatus.WAITING_REVIEW)
    second_store.save()
    write_json_atomic(
        batch_root / "batch_state.json",
        {
            "schema_version": "1.0",
            "batch_id": "batch_demo",
            "status": "waiting_review",
            "jobs": [
                {"job_id": "job_one", "input_name": "lesson-one.mp4", "status": "completed", "error": None},
                {"job_id": "job_two", "input_name": "lesson-two.mp4", "status": "waiting_review", "error": None},
            ],
        },
    )

    exit_code = main(["workspace-status", str(workspace)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "batch_demo" in out
    assert "lesson-one.mp4" in out
    assert "lesson-two.mp4" in out
    assert "translation 1/2" in out
    assert f"python cli.py batch resume --workspace {workspace} --batch batch_demo" in out


def test_prompt_text_replaces_invalid_terminal_bytes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_stdin = io.TextIOWrapper(io.BytesIO(b"quy tac chu\xc6oi\n"), encoding="utf-8", errors="strict")
    monkeypatch.setattr(sys, "stdin", raw_stdin)

    value = cli._prompt_text("  display_text", "old")

    assert value == "quy tac chu�oi"
    assert "display_text [old]:" in capsys.readouterr().out


def test_validate_returns_nonzero_for_missing_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_validate")
    CheckpointStore.create(paths.job_state_file, job_id="job_validate", input_file=Path("input/a.mp4")).save()

    exit_code = main(["validate", "--workspace", str(workspace), "--job", "job_validate"])

    assert exit_code == 1
    assert "manifest.json missing" in capsys.readouterr().out


def test_run_reports_input_error_for_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--input", "video.mp4"]) == 1
    assert "Input video does not exist" in capsys.readouterr().out


def test_resume_reports_missing_job_state(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["resume", "--job", "job_123"]) == 1
    assert "resume failed" in capsys.readouterr().out


class FakeSummary:
    def __init__(self, job_id: str = "job_1", *, status: str = "completed", workspace: str = "workspace") -> None:
        self.job_id = job_id
        self.status = status
        self.workspace = workspace

    def to_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "status": self.status, "workspace": self.workspace, "output_video": ""}


class FakeTTY:
    def __init__(self, text: str) -> None:
        self.lines = io.StringIO(text)

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self.lines.readline()


def test_resume_no_cache_is_passed_to_job_manager(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[dict[str, object]] = []

    def fake_resume(self: JobManager, workspace_dir: Path, job_id: str, **kwargs: object) -> FakeSummary:
        calls.append({"workspace_dir": workspace_dir, "job_id": job_id, **kwargs})
        return FakeSummary(job_id)

    monkeypatch.setattr(JobManager, "resume", fake_resume)

    assert main(["resume", "--workspace", "workspace", "--job", "job_1", "--from-stage", "tts", "--no-cache"]) == 0

    assert calls == [{"workspace_dir": Path("workspace"), "job_id": "job_1", "from_stage": StageName.TTS, "no_cache": True}]
    assert "job_1" in capsys.readouterr().out


def test_resume_auto_reviews_waiting_review_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    job_dir = workspace / "job_review"
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    required = {
        "schema_version": "1.0",
        "domain_profile": "calculus@1",
        "cue_set_checksum": "c" * 64,
        "status": "required",
        "review_scope": "translation_error",
        "cues": [
            {
                "cue_id": "cue_001",
                "reason": "translation_protected_span_review_required",
                "error": "protected span t² must be represented as t bình phương",
                "source_text_raw": "The graph is t squared.",
                "source_text_normalized": "The graph is t squared.",
                "display_text": "đồ thị theo thời gian bình phương",
                "spoken_text": "đồ thị theo thời gian bình phương",
                "protected_spans": [],
                "review_overrides": {
                    "source_text_normalized": "The graph is t squared.",
                    "display_text": "đồ thị theo thời gian bình phương",
                    "spoken_text": "đồ thị theo thời gian bình phương",
                    "protected_spans": [],
                },
            }
        ],
    }
    (artifacts / "review.required.json").write_text(json.dumps(required, ensure_ascii=False), encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_resume(self: JobManager, workspace_dir: Path, job_id: str, **kwargs: object) -> FakeSummary:
        calls.append({"workspace_dir": workspace_dir, "job_id": job_id, **kwargs})
        if len(calls) == 1:
            return FakeSummary(job_id, status=JobStatus.WAITING_REVIEW.value, workspace=str(job_dir))
        return FakeSummary(job_id, status=JobStatus.COMPLETED.value, workspace=str(job_dir))

    monkeypatch.setattr(JobManager, "resume", fake_resume)
    monkeypatch.setattr(sys, "stdin", FakeTTY("e\n\nĐồ thị là t².\nĐồ thị là t bình phương.\n[]\n"))

    assert main(["resume", "--workspace", str(workspace), "--job", "job_review"]) == 0

    assert len(calls) == 2
    output = capsys.readouterr().out
    assert "reason: translation_protected_span_review_required" in output
    assert "error: protected span t² must be represented as t bình phương" in output
    assert "action [e=edit, q=quit]:" in output
    locked = json.loads((artifacts / "review.locked.json").read_text(encoding="utf-8"))
    assert locked["status"] == "locked"
    assert locked["cues"][0]["review_overrides"]["display_text"] == "Đồ thị là t²."
    assert locked["cues"][0]["review_overrides"]["spoken_text"] == "Đồ thị là t bình phương."
    assert '"status": "completed"' in output


def test_batch_resume_no_cache_is_passed_to_batch_manager(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[dict[str, object]] = []

    def fake_resume(self: BatchManager, workspace_dir: Path, batch_id: str, **kwargs: object) -> FakeSummary:
        calls.append({"workspace_dir": workspace_dir, "batch_id": batch_id, **kwargs})
        return FakeSummary(batch_id)

    monkeypatch.setattr(BatchManager, "resume", fake_resume)

    assert main(["batch", "resume", "--workspace", "workspace", "--batch", "batch_1", "--from-stage", "tts", "--no-cache"]) == 0

    assert calls == [{"workspace_dir": Path("workspace"), "batch_id": "batch_1", "job_ids": None, "from_stage": StageName.TTS, "no_cache": True}]
    assert "batch_1" in capsys.readouterr().out


def test_review_interactive_writes_locked_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    job_dir = workspace / "job_review"
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    required = {
        "schema_version": "1.0",
        "domain_profile": "calculus@1",
        "cue_set_checksum": "a" * 64,
        "status": "required",
        "review_scope": "high_risk_cues",
        "cues": [
            {
                "cue_id": "cue_001",
                "reason": "asr_timeline_review_required",
                "risk_flags": ["tts_timing_density_high"],
                "source_text_raw": "Source one.",
                "source_text_normalized": "Source one.",
                "display_text": "Cue one.",
                "spoken_text": "Cue one.",
                "protected_spans": [],
                "review_overrides": {
                    "source_text_normalized": "Source one.",
                    "display_text": "Cue one.",
                    "spoken_text": "Cue one.",
                    "protected_spans": [],
                },
            },
            {
                "cue_id": "cue_002",
                "reason": "asr_timeline_review_required",
                "risk_flags": ["cue_duration_exceeds_max_for_safe_boundary"],
                "source_text_raw": "Source two.",
                "source_text_normalized": "Source two.",
                "display_text": "Cue two.",
                "spoken_text": "Cue two.",
                "protected_spans": [],
                "review_overrides": {
                    "source_text_normalized": "Source two.",
                    "display_text": "Cue two.",
                    "spoken_text": "Cue two.",
                    "protected_spans": [],
                },
            },
        ],
    }
    (artifacts / "review.required.json").write_text(json.dumps(required, ensure_ascii=False, indent=2), encoding="utf-8")

    responses = "\n".join(["", "e", "", "Đã sửa cue hai.", "Đã sửa spoken cue hai.", "[]"]) + "\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(responses))

    exit_code = main(["review", "--workspace", str(workspace), "--job", "job_review"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Reviewing 2 cues for job job_review" in captured
    assert not (artifacts / "review.lock.json").exists()
    assert (artifacts / "review.locked.json").exists()
    locked = json.loads((artifacts / "review.locked.json").read_text(encoding="utf-8"))
    assert locked["status"] == "locked"
    assert locked["cue_set_checksum"] == "a" * 64
    assert locked["cues"][0]["review_overrides"]["display_text"] == "Cue one."
    assert locked["cues"][1]["review_overrides"]["display_text"] == "Đã sửa cue hai."
    assert locked["cues"][1]["review_overrides"]["spoken_text"] == "Đã sửa spoken cue hai."


def test_review_interactive_sorts_legacy_required_cues_by_timeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    job_dir = workspace / "job_review_order"
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    required = {
        "schema_version": "1.0",
        "domain_profile": "calculus@1",
        "cue_set_checksum": "b" * 64,
        "status": "required",
        "review_scope": "high_risk_cues",
        "cues": [
            {
                "cue_id": "cue_late",
                "reason": "asr_timeline_review_required",
                "risk_flags": ["tts_timing_density_high"],
                "source_text_raw": "Late source.",
                "source_text_normalized": "Late source.",
                "display_text": "Late cue.",
                "spoken_text": "Late cue.",
                "protected_spans": [],
                "normalization_edits": [],
                "normalization_suggestions": [],
                "review_overrides": {
                    "source_text_normalized": "Late source.",
                    "display_text": "Late cue.",
                    "spoken_text": "Late cue.",
                    "protected_spans": [],
                },
            },
            {
                "cue_id": "cue_early",
                "reason": "asr_timeline_review_required",
                "risk_flags": ["tts_timing_density_high"],
                "source_text_raw": "Early source.",
                "source_text_normalized": "Early source.",
                "display_text": "Early cue.",
                "spoken_text": "Early cue.",
                "protected_spans": [],
                "normalization_edits": [],
                "normalization_suggestions": [],
                "review_overrides": {
                    "source_text_normalized": "Early source.",
                    "display_text": "Early cue.",
                    "spoken_text": "Early cue.",
                    "protected_spans": [],
                },
            },
        ],
    }
    (artifacts / "review.required.json").write_text(json.dumps(required, ensure_ascii=False), encoding="utf-8")
    (artifacts / "dubbing_cues.v2.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "cues": [
                    {"cue_id": "cue_early", "start_ms": 1000, "end_ms": 2000, "duration_ms": 1000},
                    {"cue_id": "cue_late", "start_ms": 3000, "end_ms": 4000, "duration_ms": 1000},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\n"))

    exit_code = main(["review", "--workspace", str(workspace), "--job", "job_review_order"])

    assert exit_code == 0
    locked = json.loads((artifacts / "review.locked.json").read_text(encoding="utf-8"))
    assert [cue["cue_id"] for cue in locked["cues"]] == ["cue_early", "cue_late"]
    assert locked["cues"][0]["review_overrides"]["start_ms"] == 1000
    assert locked["cues"][1]["review_overrides"]["start_ms"] == 3000


def test_run_openai_compatible_mode_reports_missing_provider_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--input", "video.mp4", "--provider-mode", "openai_compatible"]) == 1
    out = capsys.readouterr().out
    assert "provider config invalid" in out or "Input video does not exist" in out


def test_tts_from_stage_invalidation_preserves_tts_checkpoint(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path / "workspace", "job_tts_checkpoint")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_tts_checkpoint", input_file=Path("input/a.mp4"))
    manifest = ArtifactManifest.create("job_tts_checkpoint", paths.manifest_file)
    checkpoint = SegmentCheckpointStore.create(
        paths.artifact_path("tts_segments.v1.json"),
        stage="tts",
        segment_ids=["cue_001", "cue_002"],
    )
    audio_path = paths.tts_dir / "cue_001.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    checkpoint.mark("cue_001", StageStatus.COMPLETED, artifact="tts/cue_001.wav")
    checkpoint.mark("cue_002", StageStatus.FAILED, error="tts_semantic_quality_failed")
    checkpoint.save()
    manifest.record_artifact(
        name="tts_segments",
        version=1,
        path=checkpoint.path,
        created_by_stage=StageName.TTS,
        schema_version="1.0",
    )
    manifest.save()
    ctx = JobManager()._context(paths, store, manifest, None)

    JobManager()._invalidate_from(ctx, StageName.TTS, preserve_checkpoints=True)

    assert paths.artifact_path("tts_segments.v1.json").exists()
    reloaded_manifest = ArtifactManifest.load(paths.manifest_file)
    assert reloaded_manifest.get("tts_segments", 1) is not None
    reloaded_checkpoint = SegmentCheckpointStore.load(paths.artifact_path("tts_segments.v1.json"))
    assert reloaded_checkpoint.segments["cue_001"].status == StageStatus.COMPLETED
    assert reloaded_checkpoint.segments["cue_002"].status == StageStatus.FAILED


def test_tts_from_stage_no_cache_removes_tts_checkpoint_and_overrides(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path / "workspace", "job_tts_no_cache")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_tts_no_cache", input_file=Path("input/a.mp4"))
    manifest = ArtifactManifest.create("job_tts_no_cache", paths.manifest_file)
    checkpoint = SegmentCheckpointStore.create(
        paths.artifact_path("tts_segments.v1.json"),
        stage="tts",
        segment_ids=["cue_001"],
    )
    audio_path = paths.tts_dir / "cue_001.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    quality_dir = paths.raw_dir / "tts"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "cue_001.quality.json").write_text("{}", encoding="utf-8")
    checkpoint.mark("cue_001", StageStatus.COMPLETED, artifact="tts/cue_001.wav")
    checkpoint.save()
    write_json_atomic(
        paths.artifact_path("tts_interactive_overrides.v1.json"),
        {"schema_version": "1.0", "overrides": {"cue_001": {"spoken_text": "stale"}}},
    )
    manifest.record_artifact(
        name="tts_segments",
        version=1,
        path=checkpoint.path,
        created_by_stage=StageName.TTS,
        schema_version="1.0",
    )
    manifest.save()
    ctx = JobManager()._context(paths, store, manifest, None)

    JobManager()._invalidate_from(ctx, StageName.TTS, preserve_checkpoints=False, no_cache=True)

    assert not paths.artifact_path("tts_segments.v1.json").exists()
    assert not paths.artifact_path("tts_interactive_overrides.v1.json").exists()
    assert not audio_path.exists()
    assert not quality_dir.exists()
    reloaded_manifest = ArtifactManifest.load(paths.manifest_file)
    assert reloaded_manifest.get("tts_segments", 1) is None


def test_resume_no_cache_defaults_to_tts_for_completed_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_completed_no_cache")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_completed_no_cache", input_file=Path("input/a.mp4"))
    manifest = ArtifactManifest.create("job_completed_no_cache", paths.manifest_file)
    artifacts = paths.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    for stage in StageName:
        store.mark_stage(stage, StageStatus.COMPLETED)
    store.mark_job(JobStatus.COMPLETED)
    store.save()
    for name, version, filename, stage in [
        ("input_metadata", 1, "input_metadata.v1.json", StageName.JOB_INIT),
        ("audio_analysis", 1, "audio_analysis.v1.json", StageName.AUDIO_EXTRACT),
        ("segments", 1, "segments.v1.json", StageName.VAD),
        ("asr_segments", 1, "asr_segments.v1.json", StageName.ASR),
        ("transcript", 1, "transcript.v1.json", StageName.ASR),
        ("source_normalization", 1, "source_normalization.v1.json", StageName.ASR),
        ("glossary", 1, "glossary.locked.json", StageName.GLOSSARY),
        ("translation_blocks", 1, "translation_blocks.v1.json", StageName.TRANSLATION),
        ("dubbing_cues", 2, "dubbing_cues.v2.json", StageName.TRANSLATION),
        ("speech_timeline", 1, "speech_timeline.v1.json", StageName.TRANSLATION),
        ("translated", 2, "translated.v2.json", StageName.TRANSLATION),
        ("tts_segments", 1, "tts_segments.v1.json", StageName.TTS),
        ("tts_manifest", 1, "tts_manifest.v1.json", StageName.TTS),
        ("spoken_cues", 1, "spoken_cues.v1.json", StageName.TTS),
        ("output_video", 1, "dubbed.mp4", StageName.MIXING),
    ]:
        path = artifacts / filename if filename.endswith(".json") else paths.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        manifest.record_artifact(
            name=name,
            version=version,
            path=path,
            created_by_stage=stage,
            schema_version="1.0",
        )
    manifest.save()
    for audio_name in ("original.wav", "vocals.wav"):
        (paths.audio_dir / audio_name).write_bytes(b"audio")
    (paths.tts_dir / "mix.wav").write_bytes(b"mix")
    write_json_atomic(
        paths.artifact_path("tts_interactive_overrides.v1.json"),
        {"schema_version": "1.0", "overrides": {"cue_001": {"spoken_text": "stale"}}},
    )
    calls: list[StageName] = []

    def fake_execute(self: JobManager, ctx, *, start_stage: StageName, **kwargs: object):
        calls.append(start_stage)
        return FakeSummary(ctx.store.state.job_id)

    monkeypatch.setattr(JobManager, "_execute", fake_execute)

    summary = JobManager().resume(workspace, "job_completed_no_cache", no_cache=True)

    assert summary.job_id == "job_completed_no_cache"
    assert calls == [StageName.TTS]
    assert not paths.artifact_path("tts_segments.v1.json").exists()
    assert not paths.artifact_path("tts_interactive_overrides.v1.json").exists()


def test_first_invalid_stage_accepts_versioned_translation_artifacts(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path / "workspace", "job_versioned_translation")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_versioned_translation", input_file=Path("input/a.mp4"))
    manifest = ArtifactManifest.create("job_versioned_translation", paths.manifest_file)
    artifacts = paths.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    for stage in StageName:
        store.mark_stage(stage, StageStatus.COMPLETED)
    for name, version, filename, stage in [
        ("input_metadata", 1, "input_metadata.v1.json", StageName.JOB_INIT),
        ("audio_analysis", 1, "audio_analysis.v1.json", StageName.AUDIO_EXTRACT),
        ("segments", 1, "segments.v1.json", StageName.VAD),
        ("asr_segments", 1, "asr_segments.v1.json", StageName.ASR),
        ("transcript", 1, "transcript.v1.json", StageName.ASR),
        ("source_normalization", 1, "source_normalization.v1.json", StageName.ASR),
        ("glossary", 1, "glossary.locked.json", StageName.GLOSSARY),
        ("translation_blocks", 1, "translation_blocks.v1.json", StageName.TRANSLATION),
        ("dubbing_cues", 2, "dubbing_cues.v2.json", StageName.TRANSLATION),
        ("speech_timeline", 1, "speech_timeline.v1.json", StageName.TRANSLATION),
        ("translated", 2, "translated.v2.json", StageName.TRANSLATION),
        ("tts_segments", 1, "tts_segments.v1.json", StageName.TTS),
        ("tts_manifest", 1, "tts_manifest.v1.json", StageName.TTS),
        ("spoken_cues", 1, "spoken_cues.v1.json", StageName.TTS),
        ("output_video", 1, "output.mp4", StageName.MIXING),
    ]:
        path = artifacts / filename
        path.write_text('{"ok": true}', encoding="utf-8")
        manifest.record_artifact(name=name, version=version, path=path, created_by_stage=stage, schema_version="2.0" if version == 2 else "1.0")
    (paths.audio_dir / "original.wav").parent.mkdir(parents=True, exist_ok=True)
    (paths.audio_dir / "original.wav").write_bytes(b"wav")
    (paths.audio_dir / "vocals.wav").write_bytes(b"wav")
    (paths.tts_dir / "mix.wav").parent.mkdir(parents=True, exist_ok=True)
    (paths.tts_dir / "mix.wav").write_bytes(b"wav")
    ctx = JobManager()._context(paths, store, manifest, None)

    assert JobManager()._first_invalid_stage(ctx) is None


def test_job_manager_loads_resolved_config_path_for_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_config")
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  domain: coding",
                "mixing:",
                "  original_ducking_db: -31",
                "  tts_boost_db: 11",
                "  final_loudness_normalization: false",
            ]
        ),
        encoding="utf-8",
    )
    write_json_atomic(
        paths.root / "config.resolved.json",
        {
            "provider_mode": "mock",
            "domain": "ai",
            "glossary_review": False,
            "config_path": str(config_path),
        },
    )

    config = JobManager()._load_resolved_config(paths)

    assert config.project.domain == "ai"
    assert config.mixing.original_ducking_db == -31.0
    assert config.mixing.tts_boost_db == 11.0
    assert config.mixing.final_loudness_normalization is False



def test_batch_scan_filters_non_video_and_sorts_stably(tmp_path: Path) -> None:
    (tmp_path / "b.MP4").write_bytes(b"b")
    (tmp_path / "A.mov").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.mp4").write_bytes(b"nested")

    inputs = BatchManager().scan_inputs(tmp_path, [".mp4", ".mov"])

    assert [path.name for path in inputs] == ["A.mov", "b.MP4"]
