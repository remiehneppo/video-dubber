from __future__ import annotations

import json
from pathlib import Path

from dubber.core.enums import StageName
from dubber.core.models import DubberConfig, SubtitleConfig
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import run_mixing
from dubber.subtitles.ass import SubtitleCue, build_spoken_subtitle_cues, build_subtitle_cues, render_ass


def test_spoken_cues_use_exact_final_text_read_by_tts() -> None:
    cues = build_spoken_subtitle_cues(
        {"cues": [{
            "cue_id": "cue_1",
            "original_start_ms": 1000,
            "original_end_ms": 5000,
            "source_text": "Original source",
            "final_text": "Bản đã rút gọn thực sự được đọc.",
        }]}
    )

    assert cues[0].translation_text == "Bản đã rút gọn thực sự được đọc."
    assert cues[0].source_text == "Original source"


def test_build_subtitle_cues_splits_long_word_timed_segment() -> None:
    config = SubtitleConfig(max_cue_duration_ms=2_000, min_cue_duration_ms=500)
    transcript = {
        "segments": [
            {
                "segment_id": "seg_000001",
                "start_ms": 0,
                "end_ms": 5_000,
                "source_text": "One two three four five six.",
                "words": [
                    {"text": "One", "start_ms": 0, "end_ms": 500},
                    {"text": "two", "start_ms": 600, "end_ms": 1_100},
                    {"text": "three", "start_ms": 1_200, "end_ms": 1_700},
                    {"text": "four", "start_ms": 2_100, "end_ms": 2_600},
                    {"text": "five", "start_ms": 2_700, "end_ms": 3_300},
                    {"text": "six.", "start_ms": 3_400, "end_ms": 4_000},
                ],
            }
        ]
    }
    translated = {
        "segments": [
            {"segment_id": "seg_000001", "vi_text": "Một hai ba. Bốn năm sáu."},
        ]
    }

    cues = build_subtitle_cues(transcript, translated, config)

    assert len(cues) == 2
    assert cues[0].source_text == "One two three"
    assert cues[0].translation_text == "Một hai ba."
    assert cues[1].source_text == "four five six."
    assert cues[1].translation_text == "Bốn năm sáu."
    assert all(cue.end_ms - cue.start_ms <= 2_000 for cue in cues)


def test_build_subtitle_cues_keeps_source_when_translation_missing() -> None:
    cues = build_subtitle_cues(
        {
            "segments": [
                {"segment_id": "seg_000001", "start_ms": 100, "end_ms": 900, "source_text": "Hello."},
            ]
        },
        {"segments": []},
        SubtitleConfig(),
    )

    assert cues == [SubtitleCue(start_ms=100, end_ms=900, source_text="Hello.", translation_text="")]


def test_render_ass_uses_bottom_center_half_opacity_box_and_escapes_text() -> None:
    ass = render_ass(
        [SubtitleCue(start_ms=0, end_ms=1_500, source_text="A {brace}", translation_text="Xin chào")],
        SubtitleConfig(background_opacity=0.5, max_height_ratio=0.10, max_chars_per_line=20),
        video_width=1920,
        video_height=1080,
    )

    assert "BackColour" in ass
    assert "&H80000000" in ass
    assert "Alignment" in ass
    assert "Style: Bilingual" in ass
    assert r"\an2" in ass
    assert "MaxSubtitleHeight: 108" in ass
    assert "A (brace)" in ass
    assert r"\NXin chào" in ass


def test_run_mixing_skips_burn_in_when_subtitles_disabled(tmp_path: Path) -> None:
    ctx, copied_input, tts_audio, ffmpeg = _mixing_context(tmp_path, DubberConfig())

    output = run_mixing(ctx, copied_input, tts_audio, duration_ms=1_000)

    assert output == ctx.paths.output_dir / "input_vi.mp4"
    assert ffmpeg.mux_calls == [(copied_input, ctx.paths.audio_dir / "final_mix.wav", output)]
    assert ffmpeg.burn_calls == []
    assert not ctx.paths.artifact_path("subtitles.ass").exists()


def test_run_mixing_burns_in_subtitles_and_publishes_sidecar(tmp_path: Path) -> None:
    config = DubberConfig(subtitles=SubtitleConfig(enabled=True, output_sidecar=True))
    ctx, copied_input, tts_audio, ffmpeg = _mixing_context(tmp_path, config)

    output = run_mixing(ctx, copied_input, tts_audio, duration_ms=1_000)

    muxed = ctx.paths.output_dir / "input_vi.mux.mp4"
    ass_path = ctx.paths.artifact_path("subtitles.ass")
    assert output == ctx.paths.output_dir / "input_vi.mp4"
    assert ffmpeg.mux_calls == [(copied_input, ctx.paths.audio_dir / "final_mix.wav", muxed)]
    assert ffmpeg.burn_calls == [(muxed, ass_path, output)]
    assert ass_path.exists()
    assert "Source line" in ass_path.read_text(encoding="utf-8")
    assert "Dòng dịch" in ass_path.read_text(encoding="utf-8")
    assert ctx.manifest.validate_artifact("subtitle_ass", 1)
    assert ctx.manifest.validate_artifact("output_video", 1)
    assert ctx.store.state.stages[StageName.MIXING].artifact == "output/input_vi.mp4"


def _mixing_context(tmp_path: Path, config: DubberConfig) -> tuple[StageContext, Path, Path, "FakeFFmpeg"]:
    paths = WorkspacePaths.create(tmp_path, "job_subtitles")
    copied_input = paths.input_dir / "input.mp4"
    tts_audio = paths.tts_dir / "mix.wav"
    copied_input.write_bytes(b"video")
    tts_audio.parent.mkdir(parents=True, exist_ok=True)
    tts_audio.write_bytes(b"tts")
    (paths.audio_dir / "original.wav").write_bytes(b"original")
    _write_json(paths.artifact_path("segments.v1.json"), {"segments": [{"segment_id": "seg_000001"}]})
    _write_json(paths.artifact_path("glossary.locked.json"), {"terms": []})
    _write_json(
        paths.artifact_path("transcript.v1.json"),
        {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "start_ms": 0,
                    "end_ms": 900,
                    "source_text": "Source line",
                }
            ]
        },
    )
    _write_json(
        paths.artifact_path("translated.v2.json"),
        {"segments": [{"segment_id": "seg_000001", "vi_text": "Dòng dịch"}]},
    )
    _write_json(
        paths.artifact_path("spoken_cues.v1.json"),
        {"cues": [{
            "cue_id": "cue_1",
            "original_start_ms": 0,
            "original_end_ms": 900,
            "source_text": "Source line",
            "final_text": "Dòng dịch",
        }]},
    )
    store = CheckpointStore.create(paths.job_state_file, job_id="job_subtitles", input_file=Path("input/input.mp4"))
    manifest = ArtifactManifest.create("job_subtitles", paths.manifest_file)
    ffmpeg = FakeFFmpeg()
    ctx = StageContext(paths=paths, store=store, manifest=manifest, ffmpeg=ffmpeg, config=config)
    return ctx, copied_input, tts_audio, ffmpeg


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FakeFFmpeg:
    def __init__(self) -> None:
        self.mux_calls: list[tuple[Path, Path, Path]] = []
        self.burn_calls: list[tuple[Path, Path, Path]] = []

    def mix_commentary_audio(self, original_audio: Path, tts_audio: Path, output_audio: Path, **_: object) -> None:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"final audio")

    def mux_video_audio(self, input_video: Path, input_audio: Path, output_video: Path) -> None:
        self.mux_calls.append((input_video, input_audio, output_video))
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_video.write_bytes(b"muxed")

    def burn_in_subtitles(self, input_video: Path, subtitle_ass: Path, output_video: Path) -> None:
        self.burn_calls.append((input_video, subtitle_ass, output_video))
        output_video.write_bytes(b"burned")

    def duration_ms(self, input_path: Path) -> int:
        return 1_000

    def probe(self, input_path: Path) -> dict[str, object]:
        return {"streams": [{"codec_type": "video", "width": 160, "height": 90}]}
