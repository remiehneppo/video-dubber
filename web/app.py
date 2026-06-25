from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse

from dubber.core.io import read_json


def create_app(workspace_dir: str | Path = "workspace") -> FastAPI:
    workspace = Path(workspace_dir)
    app = FastAPI(title="Video Dubber Monitor")

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, list[dict[str, Any]]]:
        jobs: list[dict[str, Any]] = []
        if workspace.exists():
            for state_path in sorted(workspace.glob("*/job_state.json")):
                state = read_json(state_path)
                jobs.append(
                    {
                        "job_id": state["job_id"],
                        "status": state["status"],
                        "current_stage": state["current_stage"],
                        "updated_at": state["updated_at"],
                    }
                )
        return {"jobs": jobs}

    @app.websocket("/ws/jobs/{job_id}")
    async def job_progress(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        try:
            await websocket.send_json(_read_job_state(workspace, job_id))
        finally:
            await websocket.close()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _read_job_state(workspace, job_id)

    @app.get("/api/jobs/{job_id}/qa")
    def get_job_qa(job_id: str) -> dict[str, Any]:
        job_dir = _job_dir(workspace, job_id)
        _ensure_job_exists(job_dir)
        qa_files = sorted((job_dir / "output").glob("*.qa.json"))
        if not qa_files:
            raise HTTPException(status_code=404, detail="QA report not found")
        return read_json(qa_files[0])

    @app.get("/api/jobs/{job_id}/output")
    def get_job_output(job_id: str) -> FileResponse:
        job_dir = _job_dir(workspace, job_id)
        _ensure_job_exists(job_dir)
        output_files = sorted((job_dir / "output").glob("*_vi.mp4"))
        if not output_files:
            raise HTTPException(status_code=404, detail="Output video not found")
        return FileResponse(output_files[0], media_type="video/mp4", filename=output_files[0].name)

    return app


def _job_dir(workspace: Path, job_id: str) -> Path:
    if "/" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="Invalid job id")
    return workspace / job_id


def _ensure_job_exists(job_dir: Path) -> None:
    if not (job_dir / "job_state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")


def _read_job_state(workspace: Path, job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(workspace, job_id)
    _ensure_job_exists(job_dir)
    return read_json(job_dir / "job_state.json")


app = create_app(os.environ.get("VIDEO_DUBBER_WORKSPACE", "workspace"))
