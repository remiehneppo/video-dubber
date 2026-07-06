from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import TextIO

from dubber.core.io import read_json, write_json_atomic
from dubber.core.models import utc_now_iso
from dubber.core.paths import WorkspacePaths

OVERRIDES_FILENAME = "tts_interactive_overrides.v1.json"
MANUAL_TTS_ERROR_CODES = frozenset({
    "tts_semantic_quality_failed",
    "tts_rephrase_empty",
    "tts_duration_exceeds_max_speedup",
    "tts_rephrase_exceeds_char_limit",
})


@dataclass(frozen=True)
class TTSReviewDecision:
    action: str
    display_text: str = ""
    spoken_text: str = ""


@dataclass(frozen=True)
class TTSReviewRequest:
    paths: WorkspacePaths
    cue: dict[str, object]
    all_cues: list[dict[str, object]]
    failed_row: dict[str, object]

    @property
    def cue_id(self) -> str:
        return str(self.cue.get("cue_id") or self.cue.get("segment_id") or "")

    @property
    def previous_cue(self) -> dict[str, object] | None:
        return _neighbor_cue(self.all_cues, self.cue_id, offset=-1)

    @property
    def next_cue(self) -> dict[str, object] | None:
        return _neighbor_cue(self.all_cues, self.cue_id, offset=1)


class TerminalTTSReviewHandler:
    def __init__(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def review(self, request: TTSReviewRequest) -> TTSReviewDecision:
        if not self.stdin.isatty():
            raise RuntimeError("manual TTS review requires an interactive terminal")
        self._print_report(request)
        while True:
            choice = self._prompt("[e]dit, [s]ilence, [f]ail, [q]uit: ").strip().lower()
            if choice == "e":
                display_text = self._prompt_text(
                    "display_text",
                    str(request.cue.get("display_text") or request.cue.get("translated_text") or ""),
                )
                spoken_text = self._prompt_text(
                    "spoken_text",
                    str(request.cue.get("spoken_text") or request.cue.get("translated_text") or ""),
                )
                return TTSReviewDecision(action="edit", display_text=display_text, spoken_text=spoken_text)
            if choice == "s":
                return TTSReviewDecision(action="silence", display_text="", spoken_text="")
            if choice == "f":
                return TTSReviewDecision(action="fail")
            if choice == "q":
                return TTSReviewDecision(action="quit")
            print("Choose e, s, f, or q.", file=self.stdout)

    def _print_report(self, request: TTSReviewRequest) -> None:
        cue = request.cue
        failed = request.failed_row
        print(file=self.stdout)
        print(f"TTS manual review required for cue {request.cue_id}", file=self.stdout)
        print(f"  time: {_format_time(_as_int(cue.get('start_ms', 0)))} -> {_format_time(_as_int(cue.get('end_ms', 0)))}", file=self.stdout)
        for label, key in (
            ("source_text", "source_text"),
            ("display_text", "display_text"),
            ("spoken_text", "spoken_text"),
        ):
            value = cue.get(key)
            if value:
                print(f"  {label}: {value}", file=self.stdout)
        for label, neighbor in (("previous", request.previous_cue), ("next", request.next_cue)):
            if neighbor is not None:
                text = neighbor.get("display_text") or neighbor.get("spoken_text") or neighbor.get("source_text")
                print(f"  {label}: {neighbor.get('cue_id')} | {text}", file=self.stdout)
        attempts = _quality_attempts(failed)
        if attempts:
            last = attempts[-1] if isinstance(attempts[-1], dict) else {}
            print(
                "  metrics: "
                f"cer={last.get('cer', failed.get('cer', ''))} "
                f"token_recall={last.get('token_recall', failed.get('token_recall', ''))} "
                f"attempts={len(attempts)}",
                file=self.stdout,
            )
            transcript = last.get("asr_transcript")
            if transcript:
                print(f"  heard_asr: {transcript}", file=self.stdout)
            audio_paths = [
                str(attempt.get("audio_path"))
                for attempt in attempts
                if isinstance(attempt, dict) and attempt.get("audio_path")
            ]
            if audio_paths:
                print(f"  audio_attempts: {', '.join(audio_paths)}", file=self.stdout)
        final_text = failed.get("final_text")
        if final_text and final_text not in {cue.get("display_text"), cue.get("spoken_text")}:
            print(f"  final_text: {final_text}", file=self.stdout)
        final_error = failed.get("final_error")
        if final_error:
            print(f"  error: {final_error}", file=self.stdout)
            hint = _manual_review_hint(str(final_error))
            if hint:
                print(f"  action_needed: {hint}", file=self.stdout)
        suspicious = sorted(set(suspicious_unicode(" ".join(str(cue.get(key, "")) for key in ("display_text", "spoken_text")))))
        if suspicious:
            print(f"  suspicious_unicode: {', '.join(suspicious)}", file=self.stdout)

    def _prompt(self, message: str) -> str:
        print(message, end="", file=self.stdout)
        self.stdout.flush()
        return self._readline()

    def _prompt_text(self, label: str, current: str) -> str:
        print(f"Current {label}: {current}", file=self.stdout)
        current_has_replacement = _has_replacement_character(current)
        while True:
            suffix = "blank keeps current" if not current_has_replacement else "current has invalid characters; enter clean text"
            print(f"New {label} ({suffix}): ", end="", file=self.stdout)
            self.stdout.flush()
            value = self._readline().rstrip("\r\n")
            if value == "" and not current_has_replacement:
                return current
            if value == "":
                print(
                    f"  current {label} contains replacement characters; enter clean text instead of keeping it",
                    file=self.stdout,
                )
                continue
            if _has_replacement_character(value):
                print(
                    f"  {label} still contains the replacement character �; re-enter it using clean UTF-8 text",
                    file=self.stdout,
                )
                continue
            return value

    def _readline(self) -> str:
        stdin_buffer = getattr(self.stdin, "buffer", None)
        if stdin_buffer is not None:
            raw = stdin_buffer.readline()
            if raw == b"":
                raise RuntimeError("manual TTS review input closed")
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                encoding = getattr(self.stdin, "encoding", None) or "utf-8"
                return raw.decode(encoding, errors="replace")

        line = self.stdin.readline()
        if line == "":
            raise RuntimeError("manual TTS review input closed")
        return line


def _has_replacement_character(text: str) -> bool:
    return "�" in text


def _quality_attempts(failed: dict[str, object]) -> list[object]:
    attempts = failed.get("attempts")
    if isinstance(attempts, list):
        return attempts
    attempts = failed.get("quality_attempts")
    if isinstance(attempts, list):
        return attempts
    return []


def is_manual_tts_review_error(error: BaseException) -> bool:
    text = str(error)
    return any(code in text for code in MANUAL_TTS_ERROR_CODES)


def load_tts_interactive_overrides(paths: WorkspacePaths) -> dict[str, dict[str, object]]:
    path = paths.artifact_path(OVERRIDES_FILENAME)
    if not path.exists():
        return {}
    payload = read_json(path)
    overrides = payload.get("overrides", {}) if isinstance(payload, dict) else {}
    if isinstance(overrides, dict):
        return {str(key): dict(value) for key, value in overrides.items() if isinstance(value, dict)}
    if isinstance(overrides, list):
        return {
            str(item["cue_id"]): dict(item)
            for item in overrides
            if isinstance(item, dict) and item.get("cue_id") is not None
        }
    return {}


def save_tts_interactive_override(
    paths: WorkspacePaths,
    *,
    cue_id: str,
    action: str,
    display_text: str,
    spoken_text: str,
) -> None:
    overrides = load_tts_interactive_overrides(paths)
    overrides[cue_id] = {
        "cue_id": cue_id,
        "action": action,
        "display_text": display_text,
        "spoken_text": spoken_text,
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(
        paths.artifact_path(OVERRIDES_FILENAME),
        {"schema_version": "1.0", "overrides": overrides},
    )


def suspicious_unicode(text: str) -> list[str]:
    scripts: set[str] = set()
    for char in text:
        if char.isascii() or char.isspace() or char.isdigit():
            continue
        name = unicodedata.name(char, "")
        if name.startswith("LATIN ") or not name:
            continue
        scripts.add(name.split()[0].title())
    return sorted(scripts)


def _manual_review_hint(final_error: str) -> str:
    if "tts_duration_exceeds_max_speedup" in final_error:
        return (
            "audio is too long for this cue; choose edit and shorten spoken_text, "
            "or choose silence/fail/quit after the checkpoint is saved"
        )
    if "tts_semantic_quality_failed" in final_error:
        return "ASR heard different content; choose edit to correct display_text/spoken_text or choose silence/fail/quit"
    if "tts_rephrase_empty" in final_error:
        return "automatic shortening returned empty text; choose edit and provide a shorter spoken_text or choose silence/fail/quit"
    if "tts_rephrase_exceeds_char_limit" in final_error:
        max_chars = _error_int(final_error, "max_chars")
        actual_chars = _error_int(final_error, "actual_chars")
        if max_chars is not None and actual_chars is not None:
            return (
                f"automatic shortening is still too long; choose edit and shorten "
                f"display_text/spoken_text to {max_chars} characters or fewer "
                f"(currently {actual_chars} characters), or choose silence/fail/quit"
            )
        if max_chars is not None:
            return (
                f"automatic shortening is still too long; choose edit and shorten "
                f"display_text/spoken_text to {max_chars} characters or fewer, "
                f"or choose silence/fail/quit"
            )
        return "automatic shortening is still too long; choose edit and shorten display_text/spoken_text or choose silence/fail/quit"
    return ""


def _error_int(final_error: str, key: str) -> int | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\d+)(?:\s|$)", final_error)
    if match is None:
        return None
    return int(match.group(1))


def _neighbor_cue(all_cues: list[dict[str, object]], cue_id: str, *, offset: int) -> dict[str, object] | None:
    for index, cue in enumerate(all_cues):
        if str(cue.get("cue_id") or cue.get("segment_id") or "") == cue_id:
            neighbor_index = index + offset
            if 0 <= neighbor_index < len(all_cues):
                return all_cues[neighbor_index]
            return None
    return None


def _format_time(ms: int) -> str:
    total_seconds = max(0, ms) // 1000
    milliseconds = max(0, ms) % 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
