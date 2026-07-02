from __future__ import annotations

import json
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


def test_suspicious_unicode_flags_mixed_script_vietnamese() -> None:
    assert suspicious_unicode("đரờng") == ["Tamil"]
    assert suspicious_unicode("đường") == []


class NonTTY:
    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None

