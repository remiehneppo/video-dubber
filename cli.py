from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from dubber.core.enums import StageName
from dubber.core.io import read_json
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.pipeline.job_manager import JobManager, RunOptions


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dubber")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--workspace", default="workspace")
    run_parser.add_argument("--config")
    run_parser.add_argument("--domain")
    run_parser.add_argument("--provider-mode", choices=["mock", "openai_compatible"], default="mock")
    run_parser.add_argument("--glossary-review", action="store_true", default=False)
    run_parser.add_argument("--no-glossary-review", action="store_false", dest="glossary_review")
    run_parser.add_argument("--crash-stage", choices=[stage.value for stage in StageName])
    run_parser.add_argument("--crash-after-segments", type=int)
    run_parser.set_defaults(handler=cmd_run)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--job", required=True)
    resume_parser.add_argument("--workspace", default="workspace")
    resume_parser.set_defaults(handler=cmd_resume)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--job", required=True)
    status_parser.add_argument("--workspace", default="workspace")
    status_parser.set_defaults(handler=cmd_status)

    jobs_parser = subparsers.add_parser("jobs")
    jobs_parser.add_argument("--workspace", default="workspace")
    jobs_parser.set_defaults(handler=cmd_jobs)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--job", required=True)
    validate_parser.add_argument("--workspace", default="workspace")
    validate_parser.set_defaults(handler=cmd_validate)

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

    web_parser = subparsers.add_parser("web")
    web_parser.add_argument("--workspace", default="workspace")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8080)
    web_parser.set_defaults(handler=cmd_web)

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    try:
        summary = JobManager().run(
            RunOptions(
                input_path=Path(args.input),
                workspace_dir=Path(args.workspace),
                config_path=Path(args.config) if args.config else None,
                domain=args.domain,
                provider_mode=args.provider_mode,
                glossary_review=bool(args.glossary_review),
                crash_stage=args.crash_stage,
                crash_after_segments=args.crash_after_segments,
            )
        )
    except Exception as exc:
        print(f"run failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    try:
        summary = JobManager().resume(Path(args.workspace), args.job)
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


def _job_dir(workspace: str | Path, job_id: str) -> Path:
    return Path(workspace) / job_id


if __name__ == "__main__":
    raise SystemExit(main())
