from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from typing import TextIO

from dubber.core.io import read_json, write_json_atomic
from dubber.core.enums import JobStatus


class TerminalReviewSession:
    def __init__(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def complete_waiting_reviews(self, manager, workspace_dir: Path, summary):
        while getattr(summary, "status", "") == JobStatus.WAITING_REVIEW.value:
            job_id = str(getattr(summary, "job_id"))
            job_dir = workspace_dir / job_id
            required_path = job_dir / "artifacts" / "review.required.json"
            if not required_path.exists():
                break
            locked = self.build_locked_review(job_dir, job_id)
            if locked is None:
                raise RuntimeError("manual review cancelled")
            write_json_atomic(job_dir / "artifacts" / "review.locked.json", locked)
            summary = manager.resume(workspace_dir, job_id)
        return summary

    def build_locked_review(self, job_dir: Path, job_id: str) -> dict[str, object] | None:
        if not self.stdin.isatty():
            raise RuntimeError("manual review requires an interactive terminal")
        required_path = job_dir / "artifacts" / "review.required.json"
        required = read_json(required_path)
        if not isinstance(required, dict):
            raise RuntimeError(f"review.required.json is invalid for job {job_id}")
        cues = required.get("cues", [])
        if not isinstance(cues, list) or not cues:
            raise RuntimeError(f"No review cues found for job {job_id}")
        return self.interactive_review_lock(required, _chronological_review_cues(job_dir, cues), job_id=job_id)

    def interactive_review_lock(
        self,
        required: dict[str, object],
        cues: list[object],
        *,
        job_id: str,
    ) -> dict[str, object] | None:
        lock: dict[str, object] = {
            "schema_version": "1.0",
            "status": "locked",
            "cues": [],
        }
        for key in ("domain_profile", "review_scope", "cue_set_checksum"):
            if key in required and required[key] is not None:
                lock[key] = required[key]

        print(f"Reviewing {len(cues)} cues for job {job_id}", file=self.stdout)
        print("Press Enter to accept a suggestion, e to edit, or q to quit without saving.", file=self.stdout)
        print(file=self.stdout)
        for index, raw_cue in enumerate(cues, start=1):
            if not isinstance(raw_cue, dict):
                continue
            cue = raw_cue
            cue_id = str(cue.get("cue_id", f"cue_{index:06d}"))
            print(f"[{index}/{len(cues)}] {cue_id}", file=self.stdout)
            if "start_ms" in cue or "end_ms" in cue:
                start_ms = _review_int(cue.get("start_ms", 0))
                end_ms = _review_int(cue.get("end_ms", start_ms))
                print(f"  time: {_format_review_time(start_ms)} -> {_format_review_time(end_ms)}", file=self.stdout)
            reason = str(cue.get("reason", ""))
            if reason:
                print(f"  reason: {reason}", file=self.stdout)
            error = str(cue.get("error", ""))
            if error:
                print(f"  error: {error}", file=self.stdout)
                print("  action_needed: choose edit and correct the cue so the error condition is satisfied", file=self.stdout)
            risk_flags = cue.get("risk_flags", [])
            if isinstance(risk_flags, list) and risk_flags:
                print(f"  risk_flags: {', '.join(str(flag) for flag in risk_flags)}", file=self.stdout)
            for label, key in (
                ("source_text_raw", "source_text_raw"),
                ("source_text_normalized", "source_text_normalized"),
                ("display_text", "display_text"),
                ("spoken_text", "spoken_text"),
            ):
                value = cue.get(key, "")
                if value:
                    print(f"  {label}: {value}", file=self.stdout)
            protected_spans = cue.get("protected_spans", [])
            if isinstance(protected_spans, list) and protected_spans:
                print(f"  protected_spans: {json.dumps(protected_spans, ensure_ascii=False)}", file=self.stdout)

            overrides = _cue_review_overrides(cue)
            has_error = bool(str(cue.get("error", "")).strip())
            while True:
                prompt = "  action [e=edit, q=quit]: " if has_error else "  action [Enter=a/accept, e=edit, q=quit]: "
                action = self.prompt_line(prompt).strip().lower()
                if not has_error and action in {"", "a", "accept"}:
                    break
                if action in {"e", "edit"}:
                    print("  entering edit mode", file=self.stdout)
                    overrides = self.edit_review_overrides(overrides)
                    break
                if action in {"q", "quit"}:
                    return None
                print("  enter e or q" if has_error else "  enter a, e, or q", file=self.stdout)

            lock["cues"].append({"cue_id": cue_id, "review_overrides": overrides})
            print(file=self.stdout)
        return lock

    def edit_review_overrides(self, overrides: dict[str, object]) -> dict[str, object]:
        edited = _copy_review_overrides(overrides)
        edited["source_text_normalized"] = self.prompt_text("  source_text_normalized", str(edited.get("source_text_normalized", "")))
        edited["display_text"] = self.prompt_text("  display_text", str(edited.get("display_text", "")))
        edited["spoken_text"] = self.prompt_text("  spoken_text", str(edited.get("spoken_text", edited.get("display_text", ""))))
        edited["protected_spans"] = self.prompt_json_list("  protected_spans", edited.get("protected_spans", []))
        if "start_ms" in edited:
            edited["start_ms"] = self.prompt_int("  start_ms", edited["start_ms"])
        if "end_ms" in edited:
            edited["end_ms"] = self.prompt_int("  end_ms", edited["end_ms"])
        return edited

    def prompt_line(self, prompt: str) -> str:
        print(prompt, end="", file=self.stdout)
        self.stdout.flush()
        if self.stdin is sys.stdin and self.stdout is sys.stdout and self.stdin.isatty() and self.stdout.isatty():
            _enable_terminal_line_editing()
            try:
                return builtins.input("").rstrip("\r\n")
            except EOFError as exc:
                raise EOFError("interactive review input closed") from exc
        stdin_buffer = getattr(self.stdin, "buffer", None)
        if stdin_buffer is not None:
            raw = stdin_buffer.readline()
            if raw == b"":
                raise EOFError("interactive review input closed")
            try:
                return raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                encoding = getattr(self.stdin, "encoding", None) or "utf-8"
                return raw.decode(encoding, errors="replace").rstrip("\r\n")

        line = self.stdin.readline()
        if line == "":
            raise EOFError("interactive review input closed")
        return line.rstrip("\r\n")

    def prompt_text(self, label: str, current: str) -> str:
        prompt = f"{label} [{current}]: " if current else f"{label}: "
        value = self.prompt_line(prompt).strip()
        return current if value == "" else value

    def prompt_int(self, label: str, current: object) -> int:
        while True:
            prompt = f"{label} [{current}]: " if current is not None else f"{label}: "
            value = self.prompt_line(prompt).strip()
            if value == "":
                return int(current) if current is not None else 0
            try:
                return int(value)
            except ValueError:
                print(f"  invalid integer: {value}", file=self.stdout)

    def prompt_json_list(self, label: str, current: object) -> list[object]:
        while True:
            current_text = json.dumps(current, ensure_ascii=False) if current is not None else "[]"
            value = self.prompt_line(f"{label} [{current_text}]: ").strip()
            if value == "":
                return list(current) if isinstance(current, list) else []
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                print(f"  invalid JSON: {exc}", file=self.stdout)
                continue
            if isinstance(parsed, list):
                return parsed
            print("  value must be a JSON array", file=self.stdout)


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


def _copy_review_overrides(overrides: dict[str, object]) -> dict[str, object]:
    copied = dict(overrides)
    if "protected_spans" in copied and isinstance(copied["protected_spans"], list):
        copied["protected_spans"] = [dict(span) if isinstance(span, dict) else span for span in copied["protected_spans"]]
    return copied


def _enable_terminal_line_editing() -> None:
    try:
        import readline  # type: ignore[import-not-found]
    except ImportError:
        return
    parse_and_bind = getattr(readline, "parse_and_bind", None)
    if not callable(parse_and_bind):
        return
    parse_and_bind("set editing-mode emacs")
    parse_and_bind("set enable-keypad on")
