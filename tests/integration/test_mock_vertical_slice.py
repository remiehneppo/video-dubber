from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cli import main
from dubber.orchestrator.artifact_manifest import ArtifactManifest


def test_run_mock_vertical_slice_creates_output_video(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "short.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    exit_code = main(
        [
            "run",
            "--input",
            str(input_video),
            "--workspace",
            str(workspace),
            "--provider-mode",
            "mock",
            "--no-glossary-review",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    job_dir = workspace / summary["job_id"]
    output_video = job_dir / summary["output_video"]

    assert summary["status"] == "completed"
    assert output_video.exists()
    assert _has_audio_stream(output_video)

    state = json.loads((job_dir / "job_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["stages"]["mixing"]["status"] == "completed"

    manifest = ArtifactManifest.load(job_dir / "manifest.json")
    expected_artifacts = [
        ("input_metadata", 1),
        ("audio_analysis", 1),
        ("segments", 1),
        ("asr_segments", 1),
        ("transcript", 1),
        ("glossary", 1),
        ("translated", 1),
        ("tts_segments", 1),
        ("tts_manifest", 1),
        ("final_audio", 1),
        ("qa_report", 1),
        ("output_video", 1),
    ]
    for name, version in expected_artifacts:
        assert manifest.validate_artifact(name, version), name

    translated = json.loads((job_dir / "artifacts" / "translated.v1.json").read_text(encoding="utf-8"))
    assert translated["segments"]
    assert "translation_warnings" in translated["segments"][0]
    tts_manifest = json.loads((job_dir / "artifacts" / "tts_manifest.v1.json").read_text(encoding="utf-8"))
    assert tts_manifest["segments"][0]["alignment_action"] == "time_stretch"
    assert "raw_audio_path" in tts_manifest["segments"][0]


def test_glossary_review_pauses_then_resume_creates_output_video(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "review.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    run_exit = main(
        [
            "run",
            "--input",
            str(input_video),
            "--workspace",
            str(workspace),
            "--provider-mode",
            "mock",
            "--glossary-review",
        ]
    )

    assert run_exit == 0
    run_summary = json.loads(capsys.readouterr().out)
    job_dir = workspace / run_summary["job_id"]
    assert run_summary["status"] == "waiting_review"
    assert run_summary["output_video"] == ""
    assert (job_dir / "artifacts" / "glossary.draft.json").exists()
    assert not (job_dir / "output" / "review_vi.mp4").exists()

    draft = json.loads((job_dir / "artifacts" / "glossary.draft.json").read_text(encoding="utf-8"))
    draft["status"] = "locked"
    for term in draft["terms"]:
        term["locked"] = True
    (job_dir / "artifacts" / "glossary.locked.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resume_exit = main(["resume", "--workspace", str(workspace), "--job", run_summary["job_id"]])

    assert resume_exit == 0
    resume_summary = json.loads(capsys.readouterr().out)
    output_video = job_dir / resume_summary["output_video"]
    assert resume_summary["status"] == "completed"
    assert output_video.exists()
    assert _has_audio_stream(output_video)


def test_rerun_translation_repairs_corrupt_translated_artifact(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "rerun.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    assert (
        main(
            [
                "run",
                "--input",
                str(input_video),
                "--workspace",
                str(workspace),
                "--provider-mode",
                "mock",
                "--no-glossary-review",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    job_dir = workspace / summary["job_id"]
    translated_path = job_dir / "artifacts" / "translated.v1.json"
    translated_path.write_text('{"corrupt": true}', encoding="utf-8")

    assert main(["validate", "--workspace", str(workspace), "--job", summary["job_id"]]) == 1
    assert "translated.v1" in capsys.readouterr().out

    assert (
        main(
            [
                "rerun",
                "--workspace",
                str(workspace),
                "--job",
                summary["job_id"],
                "--stage",
                "translation",
            ]
        )
        == 0
    )
    rerun_summary = json.loads(capsys.readouterr().out)

    assert rerun_summary["status"] == "completed"
    assert main(["validate", "--workspace", str(workspace), "--job", summary["job_id"]]) == 0


def test_rerun_tts_segment_repairs_segment_checkpoint_and_output(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "segment.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    assert main(["run", "--input", str(input_video), "--workspace", str(workspace), "--provider-mode", "mock", "--no-glossary-review"]) == 0
    summary = json.loads(capsys.readouterr().out)
    job_dir = workspace / summary["job_id"]
    checkpoint_path = job_dir / "artifacts" / "tts_segments.v1.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    segment_id = checkpoint["segments"][0]["segment_id"]
    checkpoint["segments"][0]["status"] = "failed"
    checkpoint["segments"][0]["error"] = "simulated failure"
    checkpoint["done"] = 0
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    assert main(["rerun-segment", "--workspace", str(workspace), "--job", summary["job_id"], "--stage", "tts", "--segment", segment_id]) == 0
    rerun_summary = json.loads(capsys.readouterr().out)
    repaired = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert rerun_summary["status"] == "completed"
    assert repaired["done"] == repaired["total"]
    assert repaired["segments"][0]["status"] == "completed"
    assert main(["validate", "--workspace", str(workspace), "--job", summary["job_id"]]) == 0


def test_resume_recovers_from_tts_crash_injection(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "crash.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    assert main(["run", "--input", str(input_video), "--workspace", str(workspace), "--provider-mode", "mock", "--no-glossary-review", "--crash-stage", "tts", "--crash-after-segments", "0"]) == 1
    assert "simulated crash" in capsys.readouterr().out
    jobs = sorted(path.name for path in workspace.iterdir() if (path / "job_state.json").exists())
    assert len(jobs) == 1
    job_id = jobs[0]
    state = json.loads((workspace / job_id / "job_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["current_stage"] == "tts"

    assert main(["resume", "--workspace", str(workspace), "--job", job_id]) == 0
    summary = json.loads(capsys.readouterr().out)
    output_video = workspace / job_id / summary["output_video"]

    assert summary["status"] == "completed"
    assert output_video.exists()
    assert _has_audio_stream(output_video)
    assert main(["validate", "--workspace", str(workspace), "--job", job_id]) == 0


def _make_sample_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=15:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _has_audio_stream(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(result.stdout)
    return bool(data.get("streams"))
