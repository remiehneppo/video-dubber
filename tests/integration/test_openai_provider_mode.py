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
        if language == "vi":
            return ASRResult(
                text="Hãy nói về vectơ riêng.",
                confidence=1.0,
                language="vi",
                raw={"text": "Hãy nói về vectơ riêng."},
            )
        return ASRResult(
            text="Let's talk about eigenvectors.",
            confidence=0.91,
            language=language,
            raw={
                "fake": True,
                "text": "Let's talk about eigenvectors.",
                "words": [
                    {"word": "Let's", "start": 0.0, "end": 0.3},
                    {"word": "talk", "start": 0.4, "end": 0.7},
                    {"word": "about", "start": 0.8, "end": 1.0},
                    {"word": "eigenvectors.", "start": 1.1, "end": 1.6},
                ],
            },
        )


class FailingASRProvider:
    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        raise RuntimeError("asr server unavailable after retries")


class CalculusReviewASRProvider:
    async def transcribe(self, audio_path: Path, language: str) -> ASRResult:
        if language == "vi":
            return ASRResult(
                text="Độ dày là d r.",
                confidence=1.0,
                language="vi",
                raw={"text": "Độ dày là d r."},
            )
        raw = {
            "text": "Each ring has thickness doctor.",
            "language": language,
            "words": [
                {"word": "Each", "start": 0.0, "end": 0.2},
                {"word": "ring", "start": 0.3, "end": 0.5},
                {"word": "has", "start": 0.6, "end": 0.8},
                {"word": "thickness", "start": 0.9, "end": 1.2},
                {"word": "doctor.", "start": 1.3, "end": 1.6},
            ],
        }
        return ASRResult(text=str(raw["text"]), confidence=0.91, language=language, raw=raw)


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
        payload = json.loads(user_prompt)
        return {"segments": [
            {
                "segment_id": segment["segment_id"],
                "vi_text": "Hãy nói về vectơ riêng.",
                "used_terms": ["eigenvectors"],
                "length_ratio": 1.0,
                "translation_warnings": [],
            }
            for segment in payload["target_segments"]
        ]}


class CalculusReviewLLMProvider:
    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        if "extract" in system_prompt.lower():
            return {"terms": []}
        payload = json.loads(user_prompt)
        return {
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "vi_text": "Độ dày là d r.",
                    "used_terms": ["dr"],
                    "length_ratio": 1.0,
                    "translation_warnings": [],
                }
                for segment in payload["target_segments"]
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
            domain="seo",
        )
    )

    job_dir = workspace / summary.job_id
    transcript = json.loads((job_dir / "artifacts" / "transcript.v1.json").read_text(encoding="utf-8"))
    translated = json.loads((job_dir / "artifacts" / "translated.v1.json").read_text(encoding="utf-8"))
    glossary = json.loads((job_dir / "artifacts" / "glossary.locked.json").read_text(encoding="utf-8"))
    resolved = json.loads((job_dir / "config.resolved.json").read_text(encoding="utf-8"))

    assert summary.status == "completed"
    assert transcript["provider"]["type"] == "openai_compatible"
    assert transcript["segments"][0]["source_text"] == "Let's talk about eigenvectors."
    assert glossary["terms"][0]["vietnamese"] == "vectơ riêng"
    assert resolved["domain"] == "seo"
    assert resolved["provider_mode"] == "openai_compatible"
    assert translated["segments"][0]["vi_text"] == "Hãy nói về vectơ riêng."
    assert (job_dir / summary.output_video).exists()


def test_provider_mode_pauses_for_high_risk_source_review_then_resumes(tmp_path: Path) -> None:
    input_video = tmp_path / "provider-review.mp4"
    workspace = tmp_path / "workspace"
    _make_sample_video(input_video)
    providers = ProviderBundle(
        asr=CalculusReviewASRProvider(),
        llm=CalculusReviewLLMProvider(),
        tts=FakeTTSProvider(),
    )
    manager = JobManager(provider_bundle=providers)

    summary = manager.run(
        RunOptions(
            input_path=input_video,
            workspace_dir=workspace,
            provider_mode="openai_compatible",
            glossary_review=False,
            domain="mathematics",
        )
    )

    job_dir = workspace / summary.job_id
    assert summary.status == "waiting_review"
    assert summary.output_video == ""
    review_required = json.loads((job_dir / "artifacts" / "review.required.json").read_text(encoding="utf-8"))
    assert review_required["cues"][0]["source_text_raw"] == "Each ring has thickness doctor."
    assert review_required["cues"][0]["source_text_normalized"] == "Each ring has thickness dr."
    review_required["status"] = "locked"
    (job_dir / "artifacts" / "review.locked.json").write_text(
        json.dumps(review_required, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resumed = manager.resume(workspace, summary.job_id)

    assert resumed.status == "completed"
    assert (job_dir / resumed.output_video).exists()
    cues_v2 = json.loads((job_dir / "artifacts" / "dubbing_cues.v2.json").read_text(encoding="utf-8"))
    assert cues_v2["cues"][0]["display_text"] == "Độ dày là d r."
    assert cues_v2["cues"][0]["spoken_text"] == "Độ dày là d r."


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
                domain="seo",
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
