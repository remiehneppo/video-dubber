from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse

from dubber.core.io import read_json, write_json_atomic


def create_app(workspace_dir: str | Path = "workspace") -> FastAPI:
    workspace = Path(workspace_dir)
    app = FastAPI(title="Video Dubber Monitor")

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, list[dict[str, Any]]]:
        jobs: list[dict[str, Any]] = []
        if workspace.exists():
            state_paths = list(workspace.glob("*/job_state.json")) + list(workspace.glob("batch_*/jobs/*/job_state.json"))
            for state_path in sorted(state_paths):
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

    @app.get("/api/batches")
    def list_batches() -> dict[str, list[dict[str, Any]]]:
        batches = []
        if workspace.exists():
            for state_path in sorted(workspace.glob("batch_*/batch_state.json")):
                state = read_json(state_path)
                batches.append({
                    "batch_id": state["batch_id"],
                    "status": state["status"],
                    "updated_at": state["updated_at"],
                    "jobs": len(state.get("jobs", [])),
                })
        return {"batches": batches}

    @app.get("/api/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict[str, Any]:
        if "/" in batch_id or ".." in batch_id or not batch_id.startswith("batch_"):
            raise HTTPException(status_code=400, detail="Invalid batch id")
        state_path = workspace / batch_id / "batch_state.json"
        if not state_path.exists():
            raise HTTPException(status_code=404, detail="Batch not found")
        return read_json(state_path)

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

    @app.get("/api/jobs/{job_id}/review")
    def get_job_review(job_id: str) -> dict[str, Any]:
        job_dir = _job_dir(workspace, job_id)
        _ensure_job_exists(job_dir)
        artifacts_dir = job_dir / "artifacts"
        required_path = artifacts_dir / "review.required.json"
        locked_path = artifacts_dir / "review.locked.json"
        if not required_path.exists() and not locked_path.exists():
            raise HTTPException(status_code=404, detail="Review artifact not found")
        required = read_json(required_path) if required_path.exists() else None
        locked = read_json(locked_path) if locked_path.exists() else None
        return {
            "job_id": job_id,
            "status": "locked" if locked is not None else "required",
            "required": required,
            "locked": locked,
        }

    @app.put("/api/jobs/{job_id}/review")
    def lock_job_review(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job_dir = _job_dir(workspace, job_id)
        _ensure_job_exists(job_dir)
        locked = _review_locked_payload(payload)
        locked_path = job_dir / "artifacts" / "review.locked.json"
        write_json_atomic(locked_path, locked)
        return {
            "job_id": job_id,
            "status": "locked",
            "locked": locked,
        }

    return app


def _job_dir(workspace: Path, job_id: str) -> Path:
    if "/" in job_id or ".." in job_id or any(char in job_id for char in "*?[]"):
        raise HTTPException(status_code=400, detail="Invalid job id")
    direct = workspace / job_id
    if (direct / "job_state.json").exists():
        return direct
    matches = sorted(workspace.glob(f"batch_*/jobs/{job_id}"))
    if matches:
        return matches[0]
    return direct


def _ensure_job_exists(job_dir: Path) -> None:
    if not (job_dir / "job_state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")


def _read_job_state(workspace: Path, job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(workspace, job_id)
    _ensure_job_exists(job_dir)
    return read_json(job_dir / "job_state.json")


def _review_locked_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cues = payload.get("cues")
    if not isinstance(cues, list) or not cues:
        raise HTTPException(status_code=400, detail="Review lock must include a non-empty cues array")
    locked_cues: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        if not isinstance(cue, dict):
            raise HTTPException(status_code=400, detail=f"Review cue at index {index} must be an object")
        cue_id = str(cue.get("cue_id", "")).strip()
        if not cue_id:
            raise HTTPException(status_code=400, detail=f"Review cue at index {index} is missing cue_id")
        overrides = cue.get("review_overrides", cue)
        if not isinstance(overrides, dict):
            raise HTTPException(status_code=400, detail=f"Review cue {cue_id} overrides must be an object")
        allowed_overrides: dict[str, Any] = {}
        for field in ("source_text_normalized", "display_text", "spoken_text"):
            if field in overrides:
                allowed_overrides[field] = str(overrides[field])
        for field in ("start_ms", "end_ms"):
            if field in overrides:
                value = overrides[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise HTTPException(status_code=400, detail=f"Review cue {cue_id} {field} must be a non-negative integer")
                allowed_overrides[field] = value
        if "protected_spans" in overrides:
            if not isinstance(overrides["protected_spans"], list):
                raise HTTPException(status_code=400, detail=f"Review cue {cue_id} protected_spans must be an array")
            allowed_overrides["protected_spans"] = overrides["protected_spans"]
        if not allowed_overrides:
            raise HTTPException(status_code=400, detail=f"Review cue {cue_id} has no supported overrides")
        locked_cues.append({
            "cue_id": cue_id,
            "review_overrides": allowed_overrides,
        })
    return {
        "schema_version": str(payload.get("schema_version") or "1.0"),
        "status": "locked",
        "cues": locked_cues,
    }


app = create_app(os.environ.get("VIDEO_DUBBER_WORKSPACE", "workspace"))
