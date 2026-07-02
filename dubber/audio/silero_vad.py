from __future__ import annotations

import math
import struct
import wave
from urllib.request import urlretrieve
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512
DEFAULT_MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"


@dataclass(frozen=True)
class SileroVadSegment:
    start_ms: int
    end_ms: int
    split_reason: str
    risk_flags: list[str] = field(default_factory=list)
    speech_probability: float = 1.0


class SileroVadUnavailable(RuntimeError):
    pass


def detect_silero_segments(wav_path: Path, config) -> list[SileroVadSegment]:
    model_path = ensure_silero_model(
        Path(config.silero_model_path),
        model_url=str(getattr(config, "silero_model_url", DEFAULT_MODEL_URL)),
        auto_download=bool(getattr(config, "silero_auto_download", True)),
    )
    try:
        import numpy as np  # type: ignore[import-not-found]
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SileroVadUnavailable("silero_vad_unavailable: install video-dubber[silero] to use vad.mode=silero_vad") from exc

    samples, duration_ms = _read_mono_16k_float32(wav_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    probabilities = _run_model(session, np.asarray(samples, dtype=np.float32), np)
    intervals = _probabilities_to_intervals(
        probabilities,
        threshold=float(config.silero_threshold),
        window_ms=int(WINDOW_SAMPLES * 1000 / SAMPLE_RATE),
    )
    return post_process_speech_intervals(
        intervals,
        audio_duration_ms=duration_ms,
        min_speech_duration_ms=config.min_speech_duration_ms,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_padding_ms=config.speech_padding_ms,
        target_min_chunk_ms=config.target_min_chunk_ms,
        preferred_max_chunk_ms=config.preferred_max_chunk_ms,
        hard_max_chunk_ms=config.hard_max_chunk_ms,
        merge_gap_ms=config.merge_gap_ms,
    )



def ensure_silero_model(
    model_path: Path,
    *,
    model_url: str = DEFAULT_MODEL_URL,
    auto_download: bool = True,
) -> Path:
    if model_path.exists():
        return model_path
    if not auto_download:
        raise SileroVadUnavailable(f"silero_vad_unavailable: model not found: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        urlretrieve(model_url, str(temporary_path))
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise SileroVadUnavailable(f"silero_vad_unavailable: downloaded empty model from {model_url}")
        temporary_path.replace(model_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        if isinstance(exc, SileroVadUnavailable):
            raise
        raise SileroVadUnavailable(f"silero_vad_unavailable: failed to download model from {model_url}: {exc}") from exc
    return model_path


def post_process_speech_intervals(
    intervals: Iterable[tuple[int, int]],
    *,
    audio_duration_ms: int,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_padding_ms: int,
    target_min_chunk_ms: int,
    preferred_max_chunk_ms: int,
    hard_max_chunk_ms: int,
    merge_gap_ms: int,
) -> list[SileroVadSegment]:
    filtered = [
        (max(0, int(start)), min(audio_duration_ms, int(end)))
        for start, end in intervals
        if int(end) > int(start) and int(end) - int(start) >= min_speech_duration_ms
    ]
    if not filtered:
        return [SileroVadSegment(0, audio_duration_ms, "vad_fallback_full_audio", ["no_speech_detected"], 0.0)]

    merged = _merge_intervals(filtered, max(merge_gap_ms, min_silence_duration_ms))
    padded = [
        (max(0, start - speech_padding_ms), min(audio_duration_ms, end + speech_padding_ms))
        for start, end in merged
    ]
    merged = _merge_intervals(padded, merge_gap_ms)
    merged = _merge_short_context_intervals(
        merged,
        target_min_chunk_ms=target_min_chunk_ms,
        hard_max_chunk_ms=hard_max_chunk_ms,
    )

    if preferred_max_chunk_ms <= 0:
        raise ValueError("preferred_max_chunk_ms must be positive")
    if hard_max_chunk_ms <= 0:
        raise ValueError("hard_max_chunk_ms must be positive")
    return _split_merged_intervals(
        merged,
        speech_intervals=filtered,
        target_min_chunk_ms=target_min_chunk_ms,
        preferred_max_chunk_ms=min(preferred_max_chunk_ms, hard_max_chunk_ms),
        hard_max_chunk_ms=hard_max_chunk_ms,
    )


def _split_merged_intervals(
    intervals: list[tuple[int, int]],
    *,
    speech_intervals: list[tuple[int, int]],
    target_min_chunk_ms: int,
    preferred_max_chunk_ms: int,
    hard_max_chunk_ms: int,
) -> list[SileroVadSegment]:
    segments: list[SileroVadSegment] = []
    speech = sorted(speech_intervals)
    for interval_start, interval_end in intervals:
        start = interval_start
        while interval_end - start > hard_max_chunk_ms:
            earliest = start + max(1, min(target_min_chunk_ms, preferred_max_chunk_ms))
            preferred = start + preferred_max_chunk_ms
            latest = start + hard_max_chunk_ms
            gaps = [
                (left_end, right_start)
                for (_, left_end), (right_start, _) in zip(speech, speech[1:])
                if left_end >= start and right_start <= interval_end and left_end < right_start
            ]
            candidates = [
                (gap_start + gap_end) // 2
                for gap_start, gap_end in gaps
                if earliest <= (gap_start + gap_end) // 2 <= latest
            ]
            if candidates:
                split_at = min(candidates, key=lambda value: (abs(value - preferred), value))
                reason = "vad_silero_pause_split"
                risk_flags: list[str] = []
            else:
                split_at = latest
                reason = "vad_silero_hard_split"
                risk_flags = ["hard_split"]
            segments.append(SileroVadSegment(start, split_at, reason, risk_flags))
            start = split_at
        if interval_end > start:
            segments.append(SileroVadSegment(start, interval_end, "vad_silero"))
    return segments


def _merge_short_context_intervals(
    intervals: list[tuple[int, int]],
    *,
    target_min_chunk_ms: int,
    hard_max_chunk_ms: int,
) -> list[tuple[int, int]]:
    if target_min_chunk_ms <= 0 or hard_max_chunk_ms <= 0 or not intervals:
        return intervals
    merged: list[tuple[int, int]] = []
    index = 0
    while index < len(intervals):
        start, end = intervals[index]
        while (
            end - start < target_min_chunk_ms
            and index + 1 < len(intervals)
            and intervals[index + 1][1] - start <= hard_max_chunk_ms
        ):
            index += 1
            end = intervals[index][1]
        merged.append((start, end))
        index += 1
    return merged


def _merge_intervals(intervals: list[tuple[int, int]], gap_ms: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_ms:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _probabilities_to_intervals(probabilities: list[float], *, threshold: float, window_ms: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            if start is None:
                start = index * window_ms
        elif start is not None:
            intervals.append((start, index * window_ms))
            start = None
    if start is not None:
        intervals.append((start, len(probabilities) * window_ms))
    return intervals


def _read_mono_16k_float32(wav_path: Path) -> tuple[list[float], int]:
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")
    ints = struct.unpack(f"<{len(raw) // 2}h", raw)
    mono: list[float] = []
    if channels == 1:
        mono = [sample / 32768.0 for sample in ints]
    else:
        for index in range(0, len(ints), channels):
            mono.append(sum(ints[index:index + channels]) / channels / 32768.0)
    duration_ms = int(len(mono) * 1000 / sample_rate) if sample_rate else 0
    if sample_rate == SAMPLE_RATE:
        return mono, duration_ms
    return _resample_linear(mono, sample_rate, SAMPLE_RATE), duration_ms


def _resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if not samples or source_rate <= 0:
        return []
    target_count = max(1, int(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    resampled: list[float] = []
    for index in range(target_count):
        source_pos = index * ratio
        left = int(math.floor(source_pos))
        right = min(left + 1, len(samples) - 1)
        fraction = source_pos - left
        resampled.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return resampled


def _safe_dim(dim: object) -> int:
    return int(dim) if isinstance(dim, int) and dim > 0 else 1


def _run_model(session, samples, np) -> list[float]:
    input_names = [item.name for item in session.get_inputs()]
    probabilities: list[float] = []
    state = None
    context = np.zeros((1, 64), dtype=np.float32)
    for start in range(0, len(samples), WINDOW_SAMPLES):
        chunk = samples[start:start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        chunk = chunk.reshape(1, -1)
        model_input = np.concatenate([context, chunk], axis=1)
        inputs = {}
        for name in input_names:
            if name in {"input", "x"}:
                inputs[name] = model_input
            elif name == "sr":
                inputs[name] = np.array(SAMPLE_RATE, dtype=np.int64)
            elif name in {"state", "h", "c"}:
                if state is None:
                    shape = session.get_inputs()[input_names.index(name)].shape
                    inputs[name] = np.zeros([_safe_dim(dim) for dim in shape], dtype=np.float32)
                else:
                    inputs[name] = state
        outputs = session.run(None, inputs)
        probabilities.append(float(np.asarray(outputs[0]).reshape(-1)[0]))
        context = model_input[:, -64:]
        if len(outputs) > 1:
            state = outputs[1]
    return probabilities
