from __future__ import annotations

import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TTSQualityReport:
    ok: bool
    duration_ms: int
    rms: float
    peak: int
    clipped_sample_ratio: float
    max_internal_silence_ms: int
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_tts_wav(
    audio_path: Path,
    *,
    min_rms: float = 500,
    silence_rms_threshold: int = 120,
    max_internal_silence_ms: int = 2500,
    clipping_peak_threshold: int = 32760,
    max_clipped_sample_ratio: float = 0.001,
) -> TTSQualityReport:
    samples, sample_rate, channels, frame_count = _read_pcm16_samples(audio_path)
    duration_ms = int(frame_count * 1000 / sample_rate) if sample_rate else 0
    if not samples:
        return TTSQualityReport(
            ok=False,
            duration_ms=duration_ms,
            rms=0.0,
            peak=0,
            clipped_sample_ratio=0.0,
            max_internal_silence_ms=0,
            warnings=["tts_audio_near_silent"],
        )

    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    clipped_samples = sum(1 for sample in samples if abs(sample) >= clipping_peak_threshold)
    clipped_sample_ratio = clipped_samples / len(samples)
    internal_silence_ms = _max_internal_silence_ms(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_rms_threshold,
    )

    warnings: list[str] = []
    if rms < min_rms:
        warnings.append("tts_audio_near_silent")
    if internal_silence_ms > max_internal_silence_ms:
        warnings.append("tts_audio_internal_silence")
    if peak >= clipping_peak_threshold and clipped_sample_ratio > max_clipped_sample_ratio:
        warnings.append("tts_audio_clipping")

    return TTSQualityReport(
        ok=not warnings,
        duration_ms=duration_ms,
        rms=round(rms, 2),
        peak=peak,
        clipped_sample_ratio=round(clipped_sample_ratio, 6),
        max_internal_silence_ms=internal_silence_ms,
        warnings=warnings,
    )


def _read_pcm16_samples(audio_path: Path) -> tuple[list[int], int, int, int]:
    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"TTS quality gate only supports 16-bit PCM WAV: {audio_path}")
    samples = [
        int.from_bytes(frames[index:index + 2], "little", signed=True)
        for index in range(0, len(frames), 2)
    ]
    return samples, sample_rate, channels, frame_count


def _max_internal_silence_ms(
    samples: list[int],
    *,
    sample_rate: int,
    channels: int,
    silence_threshold: int,
) -> int:
    if sample_rate <= 0 or channels <= 0:
        return 0
    frame_levels = [
        max(abs(sample) for sample in samples[index:index + channels])
        for index in range(0, len(samples), channels)
    ]
    voiced_indexes = [index for index, level in enumerate(frame_levels) if level > silence_threshold]
    if len(voiced_indexes) < 2:
        return 0
    first_voiced = voiced_indexes[0]
    last_voiced = voiced_indexes[-1]
    longest = 0
    current = 0
    for level in frame_levels[first_voiced:last_voiced + 1]:
        if level <= silence_threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest * 1000 / sample_rate)
