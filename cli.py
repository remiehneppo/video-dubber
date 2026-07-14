from __future__ import annotations

import argparse
import logging
import sys
import json
import os
from pathlib import Path
from typing import Sequence

from dubber.core.enums import JobStatus, StageName
from dubber.core.io import read_json, write_json_atomic
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.pipeline.job_manager import BatchManager, BatchOptions, JobManager, RunOptions
from dubber.pipeline.review_session import TerminalReviewSession, _chronological_review_cues
from dubber.tts.interactive_review import TerminalTTSReviewHandler


def main(argv: Sequence[str] | None = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))



def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dubber")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--workspace", default="workspace")
    run_parser.add_argument("--config")
    run_parser.add_argument("--domain")
    run_parser.add_argument("--domain-profile")
    run_parser.add_argument("--provider-mode", choices=["mock", "openai_compatible"], default="mock")
    run_parser.add_argument("--glossary-review", action="store_true", default=False)
    run_parser.add_argument("--no-glossary-review", action="store_false", dest="glossary_review")
    run_parser.add_argument("--crash-stage", choices=[stage.value for stage in StageName])
    run_parser.add_argument("--crash-after-segments", type=int)
    run_parser.set_defaults(handler=cmd_run)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--job", required=True)
    resume_parser.add_argument("--workspace", default="workspace")
    resume_parser.add_argument("--from-stage", choices=[stage.value for stage in StageName])
    resume_parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="ignore reusable segment checkpoints and TTS overrides; completed jobs default to restarting from tts",
    )
    resume_parser.add_argument("--dry-run", action="store_true", default=False, help="print the resume plan without changing artifacts")
    resume_parser.set_defaults(handler=cmd_resume)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--job", required=True)
    status_parser.add_argument("--workspace", default="workspace")
    status_parser.set_defaults(handler=cmd_status)

    jobs_parser = subparsers.add_parser("jobs")
    jobs_parser.add_argument("--workspace", default="workspace")
    jobs_parser.set_defaults(handler=cmd_jobs)

    workspace_status_parser = subparsers.add_parser("workspace-status")
    workspace_status_parser.add_argument("workspace")
    workspace_status_parser.set_defaults(handler=cmd_workspace_status)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--job", required=True)
    validate_parser.add_argument("--workspace", default="workspace")
    validate_parser.set_defaults(handler=cmd_validate)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--job", required=True)
    review_parser.add_argument("--workspace", default="workspace")
    review_parser.set_defaults(handler=cmd_review)

    rerun_parser = subparsers.add_parser("rerun")
    rerun_parser.add_argument("--job", required=True)
    rerun_parser.add_argument("--workspace", default="workspace")
    rerun_parser.add_argument("--stage", required=True, choices=[stage.value for stage in StageName])
    rerun_parser.set_defaults(handler=cmd_rerun)

    rerun_segment_parser = subparsers.add_parser("rerun-segment")
    rerun_segment_parser.add_argument("--job", required=True)
    rerun_segment_parser.add_argument("--workspace", default="workspace")
    rerun_segment_parser.add_argument("--stage", required=True, choices=[stage.value for stage in StageName])
    rerun_segment_parser.add_argument("--segment", required=True)
    rerun_segment_parser.set_defaults(handler=cmd_rerun_segment)

    batch_parser = subparsers.add_parser("batch")
    batch_commands = batch_parser.add_subparsers(dest="batch_command", required=True)

    batch_run = batch_commands.add_parser("run")
    batch_run.add_argument("--input-dir", required=True)
    batch_run.add_argument("--workspace", default="workspace")
    batch_run.add_argument("--config")
    batch_run.add_argument("--domain")
    batch_run.add_argument("--domain-profile")
    batch_run.add_argument("--provider-mode", choices=["mock", "openai_compatible"], default="mock")
    batch_run.add_argument("--glossary-review", action="store_true", default=False)
    batch_run.add_argument("--no-glossary-review", action="store_false", dest="glossary_review")
    batch_run.set_defaults(handler=cmd_batch_run)

    batch_resume = batch_commands.add_parser("resume")
    batch_resume.add_argument("--batch", required=True)
    batch_resume.add_argument("--workspace", default="workspace")
    batch_resume.add_argument("--job", action="append", dest="jobs")
    batch_resume.add_argument("--from-stage", choices=[stage.value for stage in StageName])
    batch_resume.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="ignore reusable segment checkpoints and TTS overrides; completed jobs default to restarting from tts",
    )
    batch_resume.add_argument("--dry-run", action="store_true", default=False, help="print selected job resume plans without changing artifacts")
    batch_resume.set_defaults(handler=cmd_batch_resume)

    batch_review = batch_commands.add_parser("review")
    batch_review.add_argument("--batch", required=True)
    batch_review.add_argument("--workspace", default="workspace")
    batch_review.add_argument("--job", action="append", dest="jobs")
    batch_review.set_defaults(handler=cmd_batch_review)

    batch_status = batch_commands.add_parser("status")
    batch_status.add_argument("--batch", required=True)
    batch_status.add_argument("--workspace", default="workspace")
    batch_status.set_defaults(handler=cmd_batch_status)

    batch_validate = batch_commands.add_parser("validate")
    batch_validate.add_argument("--batch", required=True)
    batch_validate.add_argument("--workspace", default="workspace")
    batch_validate.set_defaults(handler=cmd_batch_validate)

    web_parser = subparsers.add_parser("web")
    web_parser.add_argument("--workspace", default="workspace")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8080)
    web_parser.set_defaults(handler=cmd_web)

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    manager = JobManager(tts_review_handler=TerminalTTSReviewHandler().review)
    try:
        summary = manager.run(
            RunOptions(
                input_path=Path(args.input),
                workspace_dir=Path(args.workspace),
                config_path=Path(args.config) if args.config else None,
                domain=args.domain,
                domain_profile=args.domain_profile,
                provider_mode=args.provider_mode,
                glossary_review=bool(args.glossary_review),
                crash_stage=args.crash_stage,
                crash_after_segments=args.crash_after_segments,
            )
        )
        summary = TerminalReviewSession().complete_waiting_reviews(manager, Path(args.workspace), summary)
    except Exception as exc:
        print(f"run failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    manager = JobManager(tts_review_handler=TerminalTTSReviewHandler().review)
    try:
        from_stage = StageName(args.from_stage) if args.from_stage else None
        if bool(getattr(args, "dry_run", False)):
            plan = manager.plan_resume(
                Path(args.workspace),
                args.job,
                from_stage=from_stage,
                no_cache=bool(args.no_cache),
            )
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        summary = manager.resume(
            Path(args.workspace),
            args.job,
            from_stage=from_stage,
            no_cache=bool(args.no_cache),
        )
        summary = TerminalReviewSession().complete_waiting_reviews(manager, Path(args.workspace), summary)
    except Exception as exc:
        print(f"resume failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_rerun(args: argparse.Namespace) -> int:
    try:
        summary = JobManager().rerun_stage(Path(args.workspace), args.job, StageName(args.stage))
    except Exception as exc:
        print(f"rerun failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_rerun_segment(args: argparse.Namespace) -> int:
    try:
        summary = JobManager().rerun_segment(Path(args.workspace), args.job, StageName(args.stage), args.segment)
    except Exception as exc:
        print(f"rerun-segment failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_batch_run(args: argparse.Namespace) -> int:
    try:
        summary = BatchManager().run(
            BatchOptions(
                input_dir=Path(args.input_dir),
                workspace_dir=Path(args.workspace),
                config_path=Path(args.config) if args.config else None,
                domain=args.domain,
                domain_profile=args.domain_profile,
                provider_mode=args.provider_mode,
                glossary_review=bool(args.glossary_review),
            )
        )
    except Exception as exc:
        print(f"batch run failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_batch_resume(args: argparse.Namespace) -> int:
    try:
        from_stage = StageName(args.from_stage) if args.from_stage else None
        if bool(getattr(args, "dry_run", False)):
            plans = BatchManager().plan_resume(
                Path(args.workspace),
                args.batch,
                job_ids=args.jobs,
                from_stage=from_stage,
                no_cache=bool(args.no_cache),
            )
            print(json.dumps(plans, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        summary = BatchManager().resume(
            Path(args.workspace),
            args.batch,
            job_ids=args.jobs,
            from_stage=from_stage,
            no_cache=bool(args.no_cache),
        )
    except Exception as exc:
        print(f"batch resume failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_batch_review(args: argparse.Namespace) -> int:
    try:
        reviewed = _review_batch_jobs(Path(args.workspace), args.batch, args.jobs)
    except Exception as exc:
        print(f"batch review failed: {exc}")
        return 1
    print(json.dumps(reviewed, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_batch_status(args: argparse.Namespace) -> int:
    try:
        summary = BatchManager().status(Path(args.workspace), args.batch)
    except Exception as exc:
        print(f"batch status failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_batch_validate(args: argparse.Namespace) -> int:
    try:
        result = BatchManager().validate(Path(args.workspace), args.batch)
    except Exception as exc:
        print(f"batch validate failed: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def cmd_web(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError:
        print("web failed: uvicorn is not installed")
        return 1
    os.environ["VIDEO_DUBBER_WORKSPACE"] = str(Path(args.workspace))
    uvicorn.run("web.app:app", host=args.host, port=args.port)
    return 0

def cmd_status(args: argparse.Namespace) -> int:
    state_path = _job_dir(args.workspace, args.job) / "job_state.json"
    if not state_path.exists():
        print(f"job_state.json missing for job {args.job}")
        return 1
    print(json.dumps(read_json(state_path), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    if not workspace.exists():
        return 0
    for job_state in sorted(workspace.glob("*/job_state.json")):
        print(job_state.parent.name)
    return 0


def cmd_workspace_status(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"workspace missing: {workspace}")
        return 1

    lines = _format_workspace_status(workspace)
    if not lines:
        print(f"No jobs found in workspace: {workspace}")
        return 0
    print("\n".join(lines))
    return 0


def _format_workspace_status(workspace: Path) -> list[str]:
    workspace = workspace.expanduser()
    lines: list[str] = [f"Workspace: {workspace}"]
    entries = _workspace_status_entries(workspace)
    if not entries:
        return []

    for index, entry in enumerate(entries):
        if index:
            lines.append("")
        if entry["type"] == "batch":
            lines.extend(_format_batch_status_entry(workspace, entry))
        else:
            lines.extend(_format_job_status_entry(workspace, entry))
    return lines


def _workspace_status_entries(workspace: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    if (workspace / "job_state.json").exists():
        entries.append({"type": "job", "root": workspace, "state": read_json(workspace / "job_state.json")})
    if (workspace / "batch_state.json").exists():
        entries.append({"type": "batch", "root": workspace, "state": read_json(workspace / "batch_state.json")})

    for child in sorted(workspace.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir():
            continue
        if (child / "batch_state.json").exists():
            entries.append({"type": "batch", "root": child, "state": read_json(child / "batch_state.json")})
        elif (child / "job_state.json").exists():
            entries.append({"type": "job", "root": child, "state": read_json(child / "job_state.json")})
    return entries


def _format_batch_status_entry(workspace: Path, entry: dict[str, object]) -> list[str]:
    root = entry["root"]
    state = entry["state"]
    if not isinstance(root, Path) or not isinstance(state, dict):
        return []
    batch_id = str(state.get("batch_id") or root.name)
    jobs = state.get("jobs", [])
    job_count = len(jobs) if isinstance(jobs, list) else 0
    status = str(state.get("status", "unknown"))
    lines = [
        f"Batch {batch_id} | status={status} | jobs={job_count}",
        f"Resume batch: python cli.py batch resume --workspace {workspace} --batch {batch_id}",
    ]
    if status == JobStatus.WAITING_REVIEW.value:
        lines.append(f"Review batch: python cli.py batch review --workspace {workspace} --batch {batch_id}")
    if not isinstance(jobs, list):
        return lines
    for item in jobs:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id", ""))
        state_path = root / "jobs" / job_id / "job_state.json"
        job_state = read_json(state_path) if state_path.exists() else {}
        video = _video_name(item.get("input_name") or item.get("input_file") or job_state.get("input_file"))
        status = str(item.get("status") or job_state.get("status") or "unknown")
        progress = _job_progress(job_state) if isinstance(job_state, dict) and job_state else "-"
        error = (
            str(item.get("error") or job_state.get("last_error") or "")
            if isinstance(job_state, dict)
            else str(item.get("error") or "")
        )
        lines.append(_format_status_row(job_id, video, status, progress, error))
    return lines


def _format_job_status_entry(workspace: Path, entry: dict[str, object]) -> list[str]:
    root = entry["root"]
    state = entry["state"]
    if not isinstance(root, Path) or not isinstance(state, dict):
        return []
    job_id = str(state.get("job_id") or root.name)
    status = str(state.get("status", "unknown"))
    lines = [
        "Jobs:",
        _format_status_row(
            job_id,
            _video_name(state.get("input_file")),
            status,
            _job_progress(state),
            str(state.get("last_error") or ""),
        ),
        f"Resume job: python cli.py resume --workspace {workspace} --job {job_id}",
    ]
    if status == JobStatus.WAITING_REVIEW.value:
        lines.append(f"Review job: python cli.py review --workspace {workspace} --job {job_id}")
    return lines


def _format_status_row(job_id: str, video: str, status: str, progress: str, error: str) -> str:
    row = f"  {job_id} | {video} | {status} | {progress}"
    if error:
        row = f"{row} | error: {error}"
    return row


def _job_progress(state: dict[str, object]) -> str:
    current_stage = str(state.get("current_stage") or "")
    stages = state.get("stages", {})
    if not current_stage or not isinstance(stages, dict):
        return "-"
    progress = stages.get(current_stage, {})
    if not isinstance(progress, dict):
        return current_stage
    done = progress.get("done")
    total = progress.get("total")
    if done is not None and total is not None:
        return f"{current_stage} {done}/{total}"
    status = progress.get("status")
    return f"{current_stage} {status}" if status else current_stage


def _video_name(value: object) -> str:
    if value is None:
        return "-"
    name = Path(str(value)).name
    return name or str(value)


def cmd_validate(args: argparse.Namespace) -> int:
    job_dir = _job_dir(args.workspace, args.job)
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json missing for job {args.job}")
        return 1

    manifest = ArtifactManifest.load(manifest_path)
    invalid = [
        artifact
        for artifact in manifest.artifacts
        if not manifest.validate_artifact(artifact.name, artifact.version)
    ]
    if invalid:
        names = ", ".join(f"{artifact.name}.v{artifact.version}" for artifact in invalid)
        print(f"Invalid artifacts: {names}")
        return 1
    print(f"Manifest valid for job {args.job}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    job_dir = _job_dir(args.workspace, args.job)
    required_path = job_dir / "artifacts" / "review.required.json"
    if not required_path.exists():
        print(f"review.required.json missing for job {args.job}")
        return 1

    required = read_json(required_path)
    if not isinstance(required, dict):
        print(f"review.required.json is invalid for job {args.job}")
        return 1

    cues = required.get("cues", [])
    if not isinstance(cues, list) or not cues:
        print(f"No review cues found for job {args.job}")
        return 1
    locked = TerminalReviewSession().interactive_review_lock(
        required,
        _chronological_review_cues(job_dir, cues),
        job_id=args.job,
    )
    if locked is None:
        print("review cancelled")
        return 1
    output_path = job_dir / "artifacts" / "review.locked.json"
    write_json_atomic(output_path, locked)
    print(json.dumps({"job_id": args.job, "output": str(output_path)}, ensure_ascii=False, sort_keys=True))
    return 0


def _review_batch_jobs(workspace: Path, batch_id: str, job_ids: list[str] | None) -> dict[str, object]:
    batch_root = workspace / batch_id
    state_path = batch_root / "batch_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Batch state not found: {batch_id}")
    state = read_json(state_path)
    jobs = state.get("jobs", []) if isinstance(state, dict) else []
    if not isinstance(jobs, list):
        raise RuntimeError(f"batch_state.json is invalid for batch {batch_id}")
    selected = [
        job for job in jobs
        if isinstance(job, dict)
        and ((job_ids is None and str(job.get("status")) == JobStatus.WAITING_REVIEW.value)
             or (job_ids is not None and str(job.get("job_id")) in job_ids))
    ]
    if job_ids is not None:
        known = {str(job.get("job_id")) for job in jobs if isinstance(job, dict)}
        missing = sorted(set(job_ids) - known)
        if missing:
            raise ValueError(f"Unknown batch job ids: {', '.join(missing)}")
    if not selected:
        return {
            "batch_id": batch_id,
            "reviewed": [],
            "next": f"python cli.py batch resume --workspace {workspace} --batch {batch_id}",
        }

    session = TerminalReviewSession()
    reviewed: list[dict[str, str]] = []
    for job in selected:
        job_id = str(job["job_id"])
        job_dir = batch_root / "jobs" / job_id
        required_path = job_dir / "artifacts" / "review.required.json"
        if not required_path.exists():
            raise FileNotFoundError(f"review.required.json missing for job {job_id}")
        locked = session.build_locked_review(job_dir, job_id)
        if locked is None:
            raise RuntimeError("manual review cancelled")
        output_path = job_dir / "artifacts" / "review.locked.json"
        write_json_atomic(output_path, locked)
        reviewed.append({"job_id": job_id, "status": "locked", "output": str(output_path)})
    return {
        "batch_id": batch_id,
        "reviewed": reviewed,
        "next": f"python cli.py batch resume --workspace {workspace} --batch {batch_id}",
    }


def _job_dir(workspace: str | Path, job_id: str) -> Path:
    return Path(workspace) / job_id


if __name__ == "__main__":
    raise SystemExit(main())
