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
    leading_silence_ms: int
    trailing_silence_ms: int
    max_internal_silence_ms: int
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_tts_wav(
    audio_path: Path,
    *,
    min_rms: float = 500,
    silence_rms_threshold: int = 120,
    max_edge_silence_ms: int = 1200,
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
            leading_silence_ms=duration_ms,
            trailing_silence_ms=duration_ms,
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
    leading_silence_ms, trailing_silence_ms = _edge_silence_ms(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_rms_threshold,
    )

    warnings: list[str] = []
    if rms < min_rms:
        warnings.append("tts_audio_near_silent")
    if leading_silence_ms > max_edge_silence_ms:
        warnings.append("tts_audio_leading_silence")
    if trailing_silence_ms > max_edge_silence_ms:
        warnings.append("tts_audio_trailing_silence")
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
        leading_silence_ms=leading_silence_ms,
        trailing_silence_ms=trailing_silence_ms,
        max_internal_silence_ms=internal_silence_ms,
        warnings=warnings,
    )


def trim_edge_silence(
    audio_path: Path,
    *,
    silence_rms_threshold: int = 120,
    retained_edge_silence_ms: int = 200,
) -> bool:
    samples, sample_rate, channels, frame_count = _read_pcm16_samples(audio_path)
    if not samples or sample_rate <= 0 or channels <= 0:
        return False
    leading_ms, trailing_ms = _edge_silence_ms(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_rms_threshold,
    )
    retained_frames = max(0, int(sample_rate * retained_edge_silence_ms / 1000))
    trim_start_frames = max(0, int(sample_rate * leading_ms / 1000) - retained_frames)
    trim_end_frames = max(0, int(sample_rate * trailing_ms / 1000) - retained_frames)
    if trim_start_frames == 0 and trim_end_frames == 0:
        return False
    start_frame = min(trim_start_frames, frame_count)
    end_frame = max(start_frame, frame_count - trim_end_frames)
    trimmed = samples[start_frame * channels:end_frame * channels]
    if not trimmed:
        return False
    _write_pcm16_samples(audio_path, trimmed, sample_rate=sample_rate, channels=channels)
    return True


def compact_excessive_internal_silence(
    audio_path: Path,
    *,
    silence_rms_threshold: int = 120,
    max_internal_silence_ms: int = 2500,
    retained_silence_ms: int = 500,
) -> bool:
    samples, sample_rate, channels, _ = _read_pcm16_samples(audio_path)
    silence_runs = _internal_silence_runs(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_rms_threshold,
    )
    excessive_runs = [
        (start_frame, end_frame)
        for start_frame, end_frame in silence_runs
        if (end_frame - start_frame) * 1000 / sample_rate > max_internal_silence_ms
    ]
    if not excessive_runs:
        return False

    retained_frames = max(0, int(sample_rate * retained_silence_ms / 1000))
    compacted: list[int] = []
    cursor_frame = 0
    for start_frame, end_frame in excessive_runs:
        run_frames = end_frame - start_frame
        keep_frames = min(run_frames, retained_frames)
        keep_before = keep_frames // 2
        keep_after = keep_frames - keep_before
        cut_start = start_frame + keep_before
        cut_end = end_frame - keep_after
        compacted.extend(samples[cursor_frame * channels:cut_start * channels])
        cursor_frame = cut_end
    compacted.extend(samples[cursor_frame * channels:])

    _write_pcm16_samples(audio_path, compacted, sample_rate=sample_rate, channels=channels)
    return True


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


def _write_pcm16_samples(audio_path: Path, samples: list[int], *, sample_rate: int, channels: int) -> None:
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


def _window_levels(
    samples: list[int],
    *,
    sample_rate: int,
    channels: int,
    silence_threshold: int,
    window_ms: int = 20,
) -> tuple[list[float], int, int]:
    total_frames = len(samples) // channels
    window_frames = max(1, int(sample_rate * window_ms / 1000))
    overall_rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) if samples else 0.0
    adaptive_threshold = max(silence_threshold, min(1000, int(overall_rms * 0.25)))
    levels: list[float] = []
    for start_frame in range(0, total_frames, window_frames):
        end_frame = min(total_frames, start_frame + window_frames)
        window_samples = samples[start_frame * channels:end_frame * channels]
        level = math.sqrt(sum(sample * sample for sample in window_samples) / len(window_samples))
        levels.append(level)
    return levels, window_frames, adaptive_threshold


def _edge_silence_ms(
    samples: list[int],
    *,
    sample_rate: int,
    channels: int,
    silence_threshold: int,
) -> tuple[int, int]:
    if sample_rate <= 0 or channels <= 0:
        return 0, 0
    window_levels, _, adaptive_threshold = _window_levels(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_threshold,
    )
    leading_windows = 0
    for level in window_levels:
        if level > adaptive_threshold:
            break
        leading_windows += 1
    trailing_windows = 0
    for level in reversed(window_levels):
        if level > adaptive_threshold:
            break
        trailing_windows += 1
    return leading_windows * 20, trailing_windows * 20


def _max_internal_silence_ms(
    samples: list[int],
    *,
    sample_rate: int,
    channels: int,
    silence_threshold: int,
) -> int:
    silence_runs = _internal_silence_runs(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_threshold,
    )
    if not silence_runs:
        return 0
    return max(int((end_frame - start_frame) * 1000 / sample_rate) for start_frame, end_frame in silence_runs)


def _internal_silence_runs(
    samples: list[int],
    *,
    sample_rate: int,
    channels: int,
    silence_threshold: int,
    window_ms: int = 20,
) -> list[tuple[int, int]]:
    if sample_rate <= 0 or channels <= 0:
        return []

    total_frames = len(samples) // channels
    window_levels, window_frames, adaptive_threshold = _window_levels(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        silence_threshold=silence_threshold,
        window_ms=window_ms,
    )
    voiced_indexes = [index for index, level in enumerate(window_levels) if level > adaptive_threshold]
    if len(voiced_indexes) < 2:
        return []

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(voiced_indexes[0], voiced_indexes[-1] + 1):
        if window_levels[index] <= adaptive_threshold:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append((run_start * window_frames, min(total_frames, index * window_frames)))
            run_start = None
    if run_start is not None:
        runs.append((run_start * window_frames, min(total_frames, (voiced_indexes[-1] + 1) * window_frames)))
    return runs
