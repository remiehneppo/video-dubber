from __future__ import annotations

import argparse
import logging
import sys
import json
import os
from pathlib import Path
from typing import Sequence

from dubber.core.enums import StageName
from dubber.core.io import read_json, write_json_atomic
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.pipeline.job_manager import BatchManager, BatchOptions, JobManager, RunOptions
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
    batch_resume.set_defaults(handler=cmd_batch_resume)

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
    try:
        summary = JobManager(tts_review_handler=TerminalTTSReviewHandler().review).run(
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
    except Exception as exc:
        print(f"run failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    try:
        summary = JobManager(tts_review_handler=TerminalTTSReviewHandler().review).resume(
            Path(args.workspace),
            args.job,
            from_stage=StageName(args.from_stage) if args.from_stage else None,
        )
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
        summary = BatchManager().resume(
            Path(args.workspace),
            args.batch,
            job_ids=args.jobs,
            from_stage=StageName(args.from_stage) if args.from_stage else None,
        )
    except Exception as exc:
        print(f"batch resume failed: {exc}")
        return 1
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
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
    lines = [
        f"Batch {batch_id} | status={state.get('status', 'unknown')} | jobs={job_count}",
        f"Resume batch: python cli.py batch resume --workspace {workspace} --batch {batch_id}",
    ]
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
    lines = [
        "Jobs:",
        _format_status_row(
            job_id,
            _video_name(state.get("input_file")),
            str(state.get("status", "unknown")),
            _job_progress(state),
            str(state.get("last_error") or ""),
        ),
        f"Resume job: python cli.py resume --workspace {workspace} --job {job_id}",
    ]
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
    cues = _chronological_review_cues(job_dir, cues)

    locked = _interactive_review_lock(required, cues, job_id=args.job)
    if locked is None:
        print("review cancelled")
        return 1
    output_path = job_dir / "artifacts" / "review.locked.json"
    write_json_atomic(output_path, locked)
    print(json.dumps({"job_id": args.job, "output": str(output_path)}, ensure_ascii=False, sort_keys=True))
    return 0


def _chronological_review_cues(job_dir: Path, cues: list[object]) -> list[object]:
    timing_by_id = _review_timing_by_cue_id(job_dir)
    enriched: list[object] = []
    for cue in cues:
        if not isinstance(cue, dict):
            enriched.append(cue)
            continue
        cue_copy = dict(cue)
        timing = timing_by_id.get(str(cue_copy.get("cue_id", "")))
        if timing is not None:
            for key in ("start_ms", "end_ms", "duration_ms"):
                cue_copy.setdefault(key, timing[key])
            overrides = cue_copy.get("review_overrides")
            if isinstance(overrides, dict):
                override_copy = dict(overrides)
                override_copy.setdefault("start_ms", timing["start_ms"])
                override_copy.setdefault("end_ms", timing["end_ms"])
                cue_copy["review_overrides"] = override_copy
        enriched.append(cue_copy)
    return sorted(enriched, key=_review_cue_sort_key)


def _review_timing_by_cue_id(job_dir: Path) -> dict[str, dict[str, int]]:
    for filename in ("dubbing_cues.v2.json",):
        path = job_dir / "artifacts" / filename
        if not path.exists():
            continue
        payload = read_json(path)
        cues = payload.get("cues", []) if isinstance(payload, dict) else []
        if not isinstance(cues, list):
            continue
        timing: dict[str, dict[str, int]] = {}
        for cue in cues:
            if not isinstance(cue, dict):
                continue
            cue_id = str(cue.get("cue_id", ""))
            if not cue_id:
                continue
            start_ms = _review_int(cue.get("start_ms", 0))
            end_ms = _review_int(cue.get("end_ms", start_ms))
            timing[cue_id] = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": _review_int(cue.get("duration_ms", end_ms - start_ms)),
            }
        if timing:
            return timing
    return {}


def _review_cue_sort_key(cue: object) -> tuple[int, int, str]:
    if not isinstance(cue, dict):
        return (0, 0, "")
    start_ms = _review_int(cue.get("start_ms", 0))
    return (
        start_ms,
        _review_int(cue.get("end_ms", start_ms)),
        str(cue.get("cue_id", "")),
    )


def _format_review_time(ms: int) -> str:
    total_seconds = max(0, ms) // 1000
    milliseconds = max(0, ms) % 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _review_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _interactive_review_lock(required: dict[str, object], cues: list[object], *, job_id: str) -> dict[str, object] | None:
    lock: dict[str, object] = {
        "schema_version": "1.0",
        "status": "locked",
        "cues": [],
    }
    for key in ("domain_profile", "review_scope", "cue_set_checksum"):
        if key in required and required[key] is not None:
            lock[key] = required[key]

    print(f"Reviewing {len(cues)} cues for job {job_id}")
    print("Press Enter to accept a suggestion, e to edit, or q to quit without saving.")
    print()
    for index, raw_cue in enumerate(cues, start=1):
        if not isinstance(raw_cue, dict):
            continue
        cue = raw_cue
        cue_id = str(cue.get("cue_id", f"cue_{index:06d}"))
        print(f"[{index}/{len(cues)}] {cue_id}")
        if "start_ms" in cue or "end_ms" in cue:
            start_ms = _review_int(cue.get("start_ms", 0))
            end_ms = _review_int(cue.get("end_ms", start_ms))
            print(f"  time: {_format_review_time(start_ms)} -> {_format_review_time(end_ms)}")
        reason = str(cue.get("reason", ""))
        if reason:
            print(f"  reason: {reason}")
        risk_flags = cue.get("risk_flags", [])
        if isinstance(risk_flags, list) and risk_flags:
            print(f"  risk_flags: {', '.join(str(flag) for flag in risk_flags)}")
        for label, key in (
            ("source_text_raw", "source_text_raw"),
            ("source_text_normalized", "source_text_normalized"),
            ("display_text", "display_text"),
            ("spoken_text", "spoken_text"),
        ):
            value = cue.get(key, "")
            if value:
                print(f"  {label}: {value}")
        protected_spans = cue.get("protected_spans", [])
        if isinstance(protected_spans, list) and protected_spans:
            print(f"  protected_spans: {json.dumps(protected_spans, ensure_ascii=False)}")

        overrides = _cue_review_overrides(cue)
        while True:
            action = _prompt_line("  action [Enter=a/accept, e=edit, q=quit]: ").strip().lower()
            if action in {"", "a", "accept"}:
                break
            if action in {"e", "edit"}:
                print("  entering edit mode")
                overrides = _edit_review_overrides(overrides)
                break
            if action in {"q", "quit"}:
                return None
            print("  enter a, e, or q")

        lock["cues"].append({"cue_id": cue_id, "review_overrides": overrides})
        print()
    return lock



def _cue_review_overrides(cue: dict[str, object]) -> dict[str, object]:
    overrides = cue.get("review_overrides")
    if isinstance(overrides, dict) and overrides:
        return _copy_review_overrides(overrides)
    result: dict[str, object] = {
        "source_text_normalized": str(cue.get("source_text_normalized", cue.get("source_text_raw", cue.get("source_text", "")))),
        "display_text": str(cue.get("display_text", "")),
        "spoken_text": str(cue.get("spoken_text", cue.get("display_text", ""))),
        "protected_spans": list(cue.get("protected_spans", []) if isinstance(cue.get("protected_spans", []), list) else []),
    }
    if "start_ms" in cue:
        result["start_ms"] = int(cue["start_ms"])
    if "end_ms" in cue:
        result["end_ms"] = int(cue["end_ms"])
    return result



def _edit_review_overrides(overrides: dict[str, object]) -> dict[str, object]:
    edited = _copy_review_overrides(overrides)
    edited["source_text_normalized"] = _prompt_text("  source_text_normalized", str(edited.get("source_text_normalized", "")))
    edited["display_text"] = _prompt_text("  display_text", str(edited.get("display_text", "")))
    edited["spoken_text"] = _prompt_text("  spoken_text", str(edited.get("spoken_text", edited.get("display_text", ""))))
    edited["protected_spans"] = _prompt_json_list("  protected_spans", edited.get("protected_spans", []))
    if "start_ms" in edited:
        edited["start_ms"] = _prompt_int("  start_ms", edited["start_ms"])
    if "end_ms" in edited:
        edited["end_ms"] = _prompt_int("  end_ms", edited["end_ms"])
    return edited



def _copy_review_overrides(overrides: dict[str, object]) -> dict[str, object]:
    copied = dict(overrides)
    if "protected_spans" in copied and isinstance(copied["protected_spans"], list):
        copied["protected_spans"] = [dict(span) if isinstance(span, dict) else span for span in copied["protected_spans"]]
    return copied



def _prompt_line(prompt: str) -> str:
    print(prompt, end="")
    sys.stdout.flush()
    stdin_buffer = getattr(sys.stdin, "buffer", None)
    if stdin_buffer is not None:
        raw = stdin_buffer.readline()
        if raw == b"":
            raise EOFError("interactive review input closed")
        encoding = getattr(sys.stdin, "encoding", None) or "utf-8"
        return raw.decode(encoding, errors="replace").rstrip("\r\n")

    line = sys.stdin.readline()
    if line == "":
        raise EOFError("interactive review input closed")
    return line.rstrip("\r\n")


def _prompt_text(label: str, current: str) -> str:
    prompt = f"{label} [{current}]: " if current else f"{label}: "
    value = _prompt_line(prompt).strip()
    return current if value == "" else value



def _prompt_int(label: str, current: object) -> int:
    while True:
        prompt = f"{label} [{current}]: " if current is not None else f"{label}: "
        value = _prompt_line(prompt).strip()
        if value == "":
            return int(current) if current is not None else 0
        try:
            return int(value)
        except ValueError:
            print(f"  invalid integer: {value}")



def _prompt_json_list(label: str, current: object) -> list[object]:
    while True:
        current_text = json.dumps(current, ensure_ascii=False) if current is not None else "[]"
        value = _prompt_line(f"{label} [{current_text}]: ").strip()
        if value == "":
            return list(current) if isinstance(current, list) else []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            print(f"  invalid JSON: {exc}")
            continue
        if isinstance(parsed, list):
            return parsed
        print("  value must be a JSON array")


def _job_dir(workspace: str | Path, job_id: str) -> Path:
    return Path(workspace) / job_id


if __name__ == "__main__":
    raise SystemExit(main())
