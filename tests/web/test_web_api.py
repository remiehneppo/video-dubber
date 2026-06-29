from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from cli import main
from web.app import create_app


def test_web_api_lists_job_status_qa_and_output(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "web.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    assert main(["run", "--input", str(input_video), "--workspace", str(workspace), "--provider-mode", "mock", "--no-glossary-review"]) == 0
    summary = json.loads(capsys.readouterr().out)
    job_id = summary["job_id"]

    client = TestClient(create_app(workspace))

    jobs = client.get("/api/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["jobs"][0]["job_id"] == job_id

    status = client.get(f"/api/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["stages"]["mixing"]["status"] == "completed"

    qa = client.get(f"/api/jobs/{job_id}/qa")
    assert qa.status_code == 200
    assert qa.json()["segments_total"] >= 1

    output = client.get(f"/api/jobs/{job_id}/output")
    assert output.status_code == 200
    assert output.headers["content-type"].startswith("video/mp4")
    assert output.content


def test_web_api_returns_404_for_missing_job(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))

    response = client.get("/api/jobs/missing")

    assert response.status_code == 404


def test_websocket_progress_returns_job_state(tmp_path: Path, capsys) -> None:
    input_video = tmp_path / "ws.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)

    assert main(["run", "--input", str(input_video), "--workspace", str(workspace), "--provider-mode", "mock", "--no-glossary-review"]) == 0
    summary = json.loads(capsys.readouterr().out)
    client = TestClient(create_app(workspace))

    with client.websocket_connect(f"/ws/jobs/{summary['job_id']}") as websocket:
        message = websocket.receive_json()

    assert message["job_id"] == summary["job_id"]
    assert message["status"] == "completed"
    assert message["stages"]["mixing"]["status"] == "completed"


def test_web_api_lists_batch_and_resolves_nested_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    batch = workspace / "batch_demo"
    job = batch / "jobs" / "job_nested"
    job.mkdir(parents=True)
    state = {
        "schema_version": "1.0",
        "batch_id": "batch_demo",
        "status": "running",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "jobs": [{"job_id": "job_nested", "status": "running"}],
    }
    (batch / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")
    (job / "job_state.json").write_text(json.dumps({
        "job_id": "job_nested",
        "status": "running",
        "current_stage": "asr",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    client = TestClient(create_app(workspace))

    assert client.get("/api/batches").json()["batches"][0]["batch_id"] == "batch_demo"
    assert client.get("/api/batches/batch_demo").json()["status"] == "running"
    assert client.get("/api/jobs/job_nested").json()["current_stage"] == "asr"


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
