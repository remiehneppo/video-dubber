from __future__ import annotations

import io
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from dubber.core.concurrency import ProviderConcurrency
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.core.models import DubberConfig, RuntimeConfig
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.pipeline import stages
from dubber.pipeline.stage_context import StageContext
from dubber.providers.factory import ProviderBundle
from dubber.tts.interactive_review import (
    TTSReviewDecision,
    TTSReviewRequest,
    TerminalTTSReviewHandler,
    is_manual_tts_review_error,
    load_tts_interactive_overrides,
    save_tts_interactive_override,
    suspicious_unicode,
)
from dubber.tts.mock import synthesize_tone_wav


class FakeFFmpeg:
    def duration_ms(self, audio_path: Path) -> int:
        return 900


class CapturingReviewHandler:
    def __init__(self, decisions: list[TTSReviewDecision]) -> None:
        self.decisions = decisions
        self.requests: list[TTSReviewRequest] = []

    def review(self, request: TTSReviewRequest) -> TTSReviewDecision:
        self.requests.append(request)
        return self.decisions.pop(0)


def _context(tmp_path: Path, cues: list[dict[str, object]]) -> StageContext:
    paths = WorkspacePaths.create(tmp_path, "job_tts_review")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_tts_review", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_tts_review", paths.manifest_file)
    store.save()
    manifest.save()
    write_json_atomic(
        paths.artifact_path("dubbing_cues.v2.json"),
        {"schema_version": "2.0", "cues": cues},
    )
    write_json_atomic(
        paths.artifact_path("speech_timeline.v1.json"),
        {"schema_version": "1.0", "speech_intervals": [], "silence_intervals": []},
    )
    config = replace(
        DubberConfig(),
        runtime=RuntimeConfig(asr_concurrency=1, llm_concurrency=1, tts_concurrency=1),
    )
    return StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        provider_mode="openai_compatible",
        provider_bundle=ProviderBundle(asr=None, llm=None, tts=None),  # type: ignore[arg-type]
        concurrency=ProviderConcurrency(config.runtime),
        config=config,
    )


def _cue(cue_id: str, *, start_ms: int, spoken_text: str) -> dict[str, object]:
    return {
        "cue_id": cue_id,
        "start_ms": start_ms,
        "end_ms": start_ms + 1000,
        "duration_ms": 1000,
        "source_text": f"source {cue_id}",
        "translated_text": spoken_text,
        "display_text": spoken_text,
        "spoken_text": spoken_text,
        "parent_segment_ids": [cue_id.replace("cue", "seg")],
    }


def _row(ctx: StageContext, segment: dict[str, object], text: str) -> dict[str, object]:
    cue_id = str(segment["segment_id"])
    raw = ctx.paths.tts_dir / f"{cue_id}.raw.wav"
    final = ctx.paths.tts_dir / f"{cue_id}.wav"
    synthesize_tone_wav(raw, 900)
    synthesize_tone_wav(final, 900)
    return {
        "segment_id": cue_id,
        "audio_path": ctx.paths.to_relative(final),
        "raw_audio_path": ctx.paths.to_relative(raw),
        "target_start_ms": int(segment["start_ms"]),
        "orig_duration_ms": int(segment["duration_ms"]),
        "tts_duration_ms": 900,
        "alignment_action": "none",
        "final_text": text,
        "semantic_metrics": {
            "expected_text": text,
            "transcript": text,
            "cer": 0.0,
            "token_recall": 1.0,
        },
    }


def test_tts_review_retries_only_failed_cue_and_persists_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(
        tmp_path,
        [
            _cue("cue_1", start_ms=0, spoken_text="pass one"),
            _cue("cue_2", start_ms=1000, spoken_text="bad text"),
        ],
    )
    attempts: list[tuple[str, str]] = []

    async def fake_produce_provider_tts_segment(**kwargs):
        segment = kwargs["segment"]
        text = str(kwargs["text"])
        cue_id = str(segment["segment_id"])
        attempts.append((cue_id, text))
        if cue_id == "cue_2" and text == "bad text":
            quality_path = ctx.paths.raw_dir / "tts" / "cue_2.quality.json"
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                quality_path,
                {
                    "status": "failed",
                    "final_text": text,
                    "attempts": [
                        {
                            "attempt": 1,
                            "audio_path": "tts/cue_2.attempt_001.wav",
                            "asr_transcript": "wrong words",
                            "cer": 0.8,
                            "token_recall": 0.2,
                        }
                    ],
                },
            )
            raise ValueError("cue_2: tts_semantic_quality_failed cer=0.8 token_recall=0.2 attempts=1")
        return _row(ctx, segment, text)

    monkeypatch.setattr(stages, "produce_provider_tts_segment", fake_produce_provider_tts_segment)
    monkeypatch.setattr(stages, "assemble_commentary_track", lambda **kwargs: synthesize_tone_wav(kwargs["output_audio"], 1000))
    handler = CapturingReviewHandler([
        TTSReviewDecision(action="edit", display_text="fixed display", spoken_text="fixed spoken")
    ])

    stages.run_tts(ctx, 2200, tts_review_handler=handler.review)

    assert attempts == [("cue_1", "pass one"), ("cue_2", "bad text"), ("cue_2", "fixed spoken")]
    assert [request.cue_id for request in handler.requests] == ["cue_2"]
    assert handler.requests[0].previous_cue["cue_id"] == "cue_1"
    assert handler.requests[0].next_cue is None
    overrides = load_tts_interactive_overrides(ctx.paths)
    assert overrides["cue_2"]["display_text"] == "fixed display"
    assert overrides["cue_2"]["spoken_text"] == "fixed spoken"
    manifest = json.loads(ctx.paths.artifact_path("tts_manifest.v1.json").read_text(encoding="utf-8"))
    assert [row["segment_id"] for row in manifest["segments"]] == ["cue_1", "cue_2"]
    assert manifest["segments"][1]["spoken_text"] == "fixed spoken"


def test_saved_tts_override_is_applied_on_later_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="bad text")])
    save_tts_interactive_override(
        ctx.paths,
        cue_id="cue_1",
        action="edit",
        display_text="saved display",
        spoken_text="saved spoken",
    )
    attempts: list[str] = []

    async def fake_produce_provider_tts_segment(**kwargs):
        attempts.append(str(kwargs["text"]))
        return _row(ctx, kwargs["segment"], str(kwargs["text"]))

    monkeypatch.setattr(stages, "produce_provider_tts_segment", fake_produce_provider_tts_segment)
    monkeypatch.setattr(stages, "assemble_commentary_track", lambda **kwargs: synthesize_tone_wav(kwargs["output_audio"], 1000))

    stages.run_tts(ctx, 1000)

    assert attempts == ["saved spoken"]
    manifest = json.loads(ctx.paths.artifact_path("tts_manifest.v1.json").read_text(encoding="utf-8"))
    assert manifest["segments"][0]["display_text"] == "saved display"
    assert manifest["segments"][0]["spoken_text"] == "saved spoken"


def test_terminal_tts_review_prefers_utf8_for_terminal_bytes(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="bad text")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={"final_error": "cue_1: tts_semantic_quality_failed"},
    )
    stdin = BinaryTTYInput(
        "e\nđạo hàm và mối quan hệ đối nghịch giữa chúng\nđạo hàm, mối quan hệ đối nghịch giữa chúng\n".encode("utf-8"),
        encoding="ascii",
    )
    stdout = CapturingTTYOutput()

    decision = TerminalTTSReviewHandler(stdin=stdin, stdout=stdout).review(request)

    assert decision.action == "edit"
    assert decision.display_text == "đạo hàm và mối quan hệ đối nghịch giữa chúng"
    assert decision.spoken_text == "đạo hàm, mối quan hệ đối nghịch giữa chúng"
    assert "�" not in decision.display_text
    assert "�" not in decision.spoken_text


def test_terminal_tts_review_reprompts_invalid_terminal_bytes(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="bad text")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={"final_error": "cue_1: tts_semantic_quality_failed"},
    )
    stdin = BinaryTTYInput(b"e\nDisplay \xc6 text\nClean display\nSpoken text\n")
    stdout = CapturingTTYOutput()

    decision = TerminalTTSReviewHandler(stdin=stdin, stdout=stdout).review(request)

    assert decision.action == "edit"
    assert decision.display_text == "Clean display"
    assert decision.spoken_text == "Spoken text"
    assert "replacement character" in stdout.text


def test_terminal_tts_review_requires_replacing_corrupt_current_text(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="m�ối quan h�ệ")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={"final_error": "cue_1: tts_semantic_quality_failed"},
    )
    stdin = FakeTTYInput("e\n\nmối quan hệ\nmối quan hệ\n")
    stdout = CapturingTTYOutput()

    decision = TerminalTTSReviewHandler(stdin=stdin, stdout=stdout).review(request)

    assert decision.action == "edit"
    assert decision.display_text == "mối quan hệ"
    assert decision.spoken_text == "mối quan hệ"
    assert "current display_text contains replacement characters" in stdout.text


def test_terminal_tts_review_uses_input_for_real_stdio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="bad text")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={"final_error": "cue_1: tts_semantic_quality_failed"},
    )
    calls: list[str] = []
    answers = iter(["e", "display", "spoken"])

    def fake_input(prompt: str = "") -> str:
        calls.append(prompt)
        return next(answers)

    readline_calls: list[str] = []
    fake_readline = types.ModuleType("readline")
    fake_readline.parse_and_bind = readline_calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "readline", fake_readline)
    monkeypatch.setattr("builtins.input", fake_input)
    stdin = RealTTYInputTrap()
    stdout = CapturingTTYOutput()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    decision = TerminalTTSReviewHandler().review(request)

    assert decision == TTSReviewDecision(action="edit", display_text="display", spoken_text="spoken")
    assert calls == ["", "", ""]
    assert stdin.buffer_read_count == 0
    assert "set editing-mode emacs" in readline_calls
    assert "set enable-keypad on" in readline_calls


def test_terminal_tts_review_requires_tty(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="bad text")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={"final_error": "cue_1: tts_semantic_quality_failed"},
    )

    with pytest.raises(RuntimeError, match="manual TTS review requires an interactive terminal"):
        TerminalTTSReviewHandler(stdin=NonTTY(), stdout=NonTTY()).review(request)


def test_duration_exceeds_max_speedup_requires_manual_review() -> None:
    error = ValueError(
        "cue_1: tts_duration_exceeds_max_speedup duration_ms=7060 "
        "source_window_ms=4580 required_speedup_ratio=1.541 max_speedup_ratio=1.200"
    )

    assert is_manual_tts_review_error(error)


def test_rephrase_exceeds_char_limit_requires_manual_review() -> None:
    error = ValueError(
        "cue_1: tts_rephrase_exceeds_char_limit max_chars=78 "
        "actual_chars=84 attempts=3"
    )

    assert is_manual_tts_review_error(error)


def test_terminal_tts_review_explains_duration_failure(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="Đoạn này quá dài để đọc")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={
            "final_text": "Đoạn này quá dài để đọc",
            "final_error": (
                "cue_1: tts_duration_exceeds_max_speedup duration_ms=7060 "
                "source_window_ms=4580 available_overflow_ms=0 target_window_ms=4580 "
                "required_speedup_ratio=1.541 max_speedup_ratio=1.200"
            ),
        },
    )
    stdin = FakeTTYInput("q\n")
    stdout = CapturingTTYOutput()

    decision = TerminalTTSReviewHandler(stdin=stdin, stdout=stdout).review(request)

    assert decision.action == "quit"
    report = stdout.text
    assert "tts_duration_exceeds_max_speedup" in report
    assert "audio is too long for this cue" in report
    assert "shorten spoken_text" in report
    assert "silence" in report


def test_terminal_tts_review_explains_char_limit_failure(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="Một câu tiếng Việt vẫn còn hơi dài")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={
            "final_text": "Một câu tiếng Việt vẫn còn hơi dài",
            "final_error": "cue_1: tts_rephrase_exceeds_char_limit max_chars=78 actual_chars=84 attempts=3",
        },
    )
    stdin = FakeTTYInput("q\n")
    stdout = CapturingTTYOutput()

    decision = TerminalTTSReviewHandler(stdin=stdin, stdout=stdout).review(request)

    assert decision.action == "quit"
    report = stdout.text
    assert "tts_rephrase_exceeds_char_limit" in report
    assert "shorten display_text/spoken_text to 78 characters or fewer" in report
    assert "currently 84 characters" in report


def test_terminal_tts_review_prints_heard_asr_from_quality_attempts(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [_cue("cue_1", start_ms=0, spoken_text="d sin của h")])
    request = TTSReviewRequest(
        paths=ctx.paths,
        cue=dict(ctx.artifact_json("dubbing_cues.v2.json")["cues"][0], segment_id="cue_1"),
        all_cues=ctx.artifact_json("dubbing_cues.v2.json")["cues"],
        failed_row={
            "final_error": "cue_1: tts_semantic_quality_failed cer=0.2 token_recall=0.7 attempts=2",
            "quality_attempts": [
                {
                    "attempt": 1,
                    "audio_path": "raw/tts/cue_1.attempt_001.wav",
                    "asr_transcript": "d sinh của hát",
                    "cer": 0.4,
                    "token_recall": 0.6,
                },
                {
                    "attempt": 2,
                    "audio_path": "raw/tts/cue_1.attempt_002.wav",
                    "asr_transcript": "d sinh của hát này",
                    "cer": 0.2,
                    "token_recall": 0.7,
                },
            ],
        },
    )
    stdin = FakeTTYInput("q\n")
    stdout = CapturingTTYOutput()

    decision = TerminalTTSReviewHandler(stdin=stdin, stdout=stdout).review(request)

    assert decision.action == "quit"
    report = stdout.text
    assert "metrics: cer=0.2 token_recall=0.7 attempts=2" in report
    assert "heard_asr: d sinh của hát này" in report
    assert "audio_attempts: raw/tts/cue_1.attempt_001.wav, raw/tts/cue_1.attempt_002.wav" in report


def test_suspicious_unicode_flags_mixed_script_vietnamese() -> None:
    assert suspicious_unicode("đரờng") == ["Tamil"]
    assert suspicious_unicode("đường") == []


class BinaryTTYInput:
    def __init__(self, data: bytes, *, encoding: str = "utf-8") -> None:
        self.buffer = io.BytesIO(data)
        self.encoding = encoding

    def isatty(self) -> bool:
        return True


class NonTTY:
    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


class _TrapBuffer:
    def __init__(self, owner: "RealTTYInputTrap") -> None:
        self.owner = owner

    def readline(self) -> bytes:
        self.owner.buffer_read_count += 1
        raise AssertionError("real terminal input should use input(), not stdin.buffer.readline()")


class RealTTYInputTrap:
    def __init__(self) -> None:
        self.buffer_read_count = 0
        self.buffer = _TrapBuffer(self)
        self.encoding = "utf-8"

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        raise AssertionError("real terminal input should use input(), not stdin.readline()")


class FakeTTYInput:
    def __init__(self, text: str) -> None:
        self.lines = text.splitlines(keepends=True)

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        if not self.lines:
            return ""
        return self.lines.pop(0)


class CapturingTTYOutput:
    def __init__(self) -> None:
        self.text = ""

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.text += text
        return len(text)

    def flush(self) -> None:
        return None
