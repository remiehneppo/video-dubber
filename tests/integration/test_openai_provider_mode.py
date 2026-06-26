from __future__ import annotations

import json
import subprocess
from pathlib import Path

from dubber.pipeline.job_manager import JobManager, RunOptions
from dubber.providers.base import ASRResult, TTSResult
from dubber.providers.factory import ProviderBundle
from dubber.tts.mock import synthesize_tone_wav


class FakeASRProvider:
    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        return ASRResult(
            text="Let's talk about eigenvectors.",
            confidence=0.91,
            language=language,
            raw={"fake": True},
        )


class FailingASRProvider:
    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        raise RuntimeError("asr server unavailable after retries")


class FakeLLMProvider:
    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        if "terminology" in user_prompt.lower() or "extract" in system_prompt.lower():
            return {
                "terms": [
                    {
                        "original": "eigenvectors",
                        "vietnamese": "vectơ riêng",
                        "category": "math_term",
                        "confidence": 0.99,
                        "source_segments": ["seg_000001"],
                        "notes": "",
                    }
                ]
            }
        return {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "vi_text": "Hãy nói về vectơ riêng.",
                    "used_terms": ["eigenvectors"],
                }
            ]
        }


class FakeTTSProvider:
    async def synthesize(self, text: str, voice: str, output_path: Path) -> TTSResult:
        synthesize_tone_wav(output_path, 900)
        return TTSResult(audio_path=output_path, duration_ms=900, provider_metadata={"fake": True})


def test_openai_compatible_provider_mode_runs_with_injected_providers(tmp_path: Path) -> None:
    input_video = tmp_path / "provider.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)
    providers = ProviderBundle(asr=FakeASRProvider(), llm=FakeLLMProvider(), tts=FakeTTSProvider())

    summary = JobManager(provider_bundle=providers).run(
        RunOptions(
            input_path=input_video,
            workspace_dir=workspace,
            provider_mode="openai_compatible",
            glossary_review=False,
        )
    )

    job_dir = workspace / summary.job_id
    transcript = json.loads((job_dir / "artifacts" / "transcript.v1.json").read_text(encoding="utf-8"))
    translated = json.loads((job_dir / "artifacts" / "translated.v1.json").read_text(encoding="utf-8"))
    glossary = json.loads((job_dir / "artifacts" / "glossary.locked.json").read_text(encoding="utf-8"))

    assert summary.status == "completed"
    assert transcript["provider"]["type"] == "openai_compatible"
    assert transcript["segments"][0]["source_text"] == "Let's talk about eigenvectors."
    assert glossary["terms"][0]["vietnamese"] == "vectơ riêng"
    assert translated["segments"][0]["vi_text"] == "Hãy nói về vectơ riêng."
    assert (job_dir / summary.output_video).exists()


def test_provider_failure_marks_job_failed_and_saves_state(tmp_path: Path) -> None:
    input_video = tmp_path / "provider-fail.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)
    providers = ProviderBundle(asr=FailingASRProvider(), llm=FakeLLMProvider(), tts=FakeTTSProvider())

    try:
        JobManager(provider_bundle=providers).run(
            RunOptions(
                input_path=input_video,
                workspace_dir=workspace,
                provider_mode="openai_compatible",
                glossary_review=False,
            )
        )
    except RuntimeError as exc:
        assert "asr server unavailable after retries" in str(exc)
    else:
        raise AssertionError("Expected provider failure")

    job_dirs = [path for path in workspace.iterdir() if (path / "job_state.json").exists()]
    assert len(job_dirs) == 1
    state = json.loads((job_dirs[0] / "job_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["current_stage"] == "asr"
    assert "asr server unavailable after retries" in state["last_error"]


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
