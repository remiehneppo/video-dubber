from __future__ import annotations

import wave
from pathlib import Path

from dubber.tts.audio_quality import analyze_tts_wav
from dubber.tts.mock import synthesize_silence_wav, synthesize_tone_wav


def _write_samples(path: Path, samples: list[int], *, sample_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


def test_tone_audio_passes_quality_gate(tmp_path: Path) -> None:
    audio = tmp_path / "tone.wav"
    synthesize_tone_wav(audio, 500)

    report = analyze_tts_wav(audio)

    assert report.ok is True
    assert report.duration_ms == 500
    assert report.rms >= 500
    assert report.warnings == []


def test_near_silence_fails_quality_gate(tmp_path: Path) -> None:
    audio = tmp_path / "silence.wav"
    synthesize_silence_wav(audio, 500)

    report = analyze_tts_wav(audio)

    assert report.ok is False
    assert "tts_audio_near_silent" in report.warnings


def test_internal_silence_fails_quality_gate(tmp_path: Path) -> None:
    audio = tmp_path / "internal_silence.wav"
    _write_samples(audio, [1000] * 100 + [0] * 3000 + [1000] * 100, sample_rate=1000)

    report = analyze_tts_wav(audio, max_internal_silence_ms=2500, silence_rms_threshold=120)

    assert report.ok is False
    assert report.max_internal_silence_ms == 3000
    assert "tts_audio_internal_silence" in report.warnings


def test_clipping_fails_quality_gate(tmp_path: Path) -> None:
    audio = tmp_path / "clipped.wav"
    _write_samples(audio, [32767] * 20 + [1000] * 980, sample_rate=1000)

    report = analyze_tts_wav(audio, max_clipped_sample_ratio=0.001)

    assert report.ok is False
    assert report.clipped_sample_ratio > 0.001
    assert "tts_audio_clipping" in report.warnings
