from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VadConfig:
    mode: str = "energy_segments"
    frame_ms: int = 100
    threshold_ratio: float = 0.08
    min_duration_ms: int = 900
    max_duration_ms: int = 60_000
    min_speech_duration_ms: int = 700
    target_min_chunk_ms: int = 20_000
    preferred_max_chunk_ms: int = 45_000
    hard_max_chunk_ms: int = 90_000
    silence_merge_threshold_ms: int = 2_500
    context_padding_ms: int = 0
    soft_split_allowed: bool = True


@dataclass(frozen=True)
class VadSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    duration_ms: int
    silence_before_ms: int
    silence_after_ms: int
    speech_probability: float
    split_reason: str
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "silence_before_ms": self.silence_before_ms,
            "silence_after_ms": self.silence_after_ms,
            "speech_probability": self.speech_probability,
            "split_reason": self.split_reason,
            "risk_flags": self.risk_flags,
        }


@dataclass(frozen=True)
class _Frame:
    start_ms: int
    end_ms: int
    rms: int


@dataclass(frozen=True)
class _Interval:
    start_ms: int
    end_ms: int
    split_reason: str
    risk_flags: list[str]


def detect_segments(wav_path: Path, config: VadConfig | None = None) -> list[VadSegment]:
    config = config or VadConfig()
    frames, duration_ms = _read_frames(wav_path, config.frame_ms)
    if not frames:
        raise ValueError(f"WAV contains no audio frames: {wav_path}")

    max_rms = max(frame.rms for frame in frames)
    threshold = 0.0
    if max_rms <= 0:
        intervals = [_Interval(0, duration_ms, "vad_fallback_full_audio", ["no_speech_detected"])]
    else:
        threshold = max_rms * config.threshold_ratio
        intervals = _speech_intervals(frames, threshold)
        intervals = _merge_intervals(intervals, config.silence_merge_threshold_ms)
        intervals = _pad_intervals(intervals, duration_ms, config.context_padding_ms)
        minimum_duration_ms = config.min_speech_duration_ms if config.mode == "asr_context_chunks" else config.min_duration_ms
        intervals = [
            interval
            for interval in intervals
            if interval.end_ms - interval.start_ms >= minimum_duration_ms
        ]
        if config.mode == "asr_context_chunks":
            intervals = _merge_short_context_chunks(intervals, config.target_min_chunk_ms)
        if not intervals:
            intervals = [_Interval(0, duration_ms, "vad_fallback_full_audio", ["no_speech_detected"])]

    if config.mode == "asr_context_chunks":
        split_intervals = _split_context_intervals(intervals, frames, threshold, config)
    else:
        split_intervals = _split_long_intervals(intervals, config.max_duration_ms, config.soft_split_allowed)
    return [
        VadSegment(
            segment_id=f"seg_{index:06d}",
            start_ms=interval.start_ms,
            end_ms=interval.end_ms,
            duration_ms=interval.end_ms - interval.start_ms,
            silence_before_ms=interval.start_ms - split_intervals[index - 2].end_ms if index > 1 else interval.start_ms,
            silence_after_ms=(
                split_intervals[index].start_ms - interval.end_ms
                if index < len(split_intervals)
                else max(0, duration_ms - interval.end_ms)
            ),
            speech_probability=0.0 if "no_speech_detected" in interval.risk_flags else 1.0,
            split_reason=interval.split_reason,
            risk_flags=interval.risk_flags,
        )
        for index, interval in enumerate(split_intervals, start=1)
    ]


def _read_frames(wav_path: Path, frame_ms: int) -> tuple[list[_Frame], int]:
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    frames: list[_Frame] = []
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        total_frames = wav.getnframes()
        samples_per_frame = max(1, int(sample_rate * frame_ms / 1000))
        frame_index = 0
        while True:
            raw = wav.readframes(samples_per_frame)
            if not raw:
                break
            start_ms = int(frame_index * samples_per_frame * 1000 / sample_rate)
            end_ms = int((frame_index * samples_per_frame + len(raw) / sample_width / channels) * 1000 / sample_rate)
            frames.append(_Frame(start_ms=start_ms, end_ms=end_ms, rms=_rms(raw, sample_width, channels)))
            frame_index += 1
        duration_ms = int(total_frames * 1000 / sample_rate)
    return frames, duration_ms


def _rms(raw: bytes, sample_width: int, channels: int) -> int:
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    if not samples:
        return 0
    if channels > 1:
        samples = samples[::channels]
    square_mean = sum(sample * sample for sample in samples) / len(samples)
    return int(math.sqrt(square_mean))


def _speech_intervals(frames: list[_Frame], threshold: float) -> list[_Interval]:
    intervals: list[_Interval] = []
    current_start: int | None = None
    current_end: int | None = None
    for frame in frames:
        if frame.rms >= threshold:
            if current_start is None:
                current_start = frame.start_ms
            current_end = frame.end_ms
        elif current_start is not None and current_end is not None:
            intervals.append(_Interval(current_start, current_end, "vad_energy", []))
            current_start = None
            current_end = None
    if current_start is not None and current_end is not None:
        intervals.append(_Interval(current_start, current_end, "vad_energy", []))
    return intervals


def _merge_intervals(intervals: list[_Interval], silence_merge_threshold_ms: int) -> list[_Interval]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for interval in intervals[1:]:
        previous = merged[-1]
        gap = interval.start_ms - previous.end_ms
        if gap <= silence_merge_threshold_ms:
            merged[-1] = _Interval(previous.start_ms, interval.end_ms, previous.split_reason, previous.risk_flags)
        else:
            merged.append(interval)
    return merged


def _pad_intervals(intervals: list[_Interval], duration_ms: int, padding_ms: int) -> list[_Interval]:
    if padding_ms <= 0:
        return intervals
    padded: list[_Interval] = []
    for interval in intervals:
        start = max(0, interval.start_ms - padding_ms)
        end = min(duration_ms, interval.end_ms + padding_ms)
        padded.append(_Interval(start, end, interval.split_reason, interval.risk_flags))
    return _merge_intervals(padded, padding_ms)


def _merge_short_context_chunks(intervals: list[_Interval], target_min_chunk_ms: int) -> list[_Interval]:
    if target_min_chunk_ms <= 0 or not intervals:
        return intervals
    merged: list[_Interval] = []
    index = 0
    while index < len(intervals):
        current = intervals[index]
        changed = False
        while current.end_ms - current.start_ms < target_min_chunk_ms and index + 1 < len(intervals):
            index += 1
            next_interval = intervals[index]
            current = _Interval(
                current.start_ms,
                next_interval.end_ms,
                "vad_context_merge",
                _merge_risk_flags(current.risk_flags, next_interval.risk_flags),
            )
            changed = True
        if changed and "short_context_merged" not in current.risk_flags:
            current = _Interval(current.start_ms, current.end_ms, current.split_reason, current.risk_flags + ["short_context_merged"])
        merged.append(current)
        index += 1
    return merged


def _split_context_intervals(
    intervals: list[_Interval],
    frames: list[_Frame],
    threshold: float,
    config: VadConfig,
) -> list[_Interval]:
    hard_max_ms = min(config.hard_max_chunk_ms, config.max_duration_ms)
    preferred_max_ms = min(config.preferred_max_chunk_ms, hard_max_ms)
    if hard_max_ms <= 0:
        raise ValueError("hard_max_chunk_ms must be positive")
    if preferred_max_ms <= 0:
        raise ValueError("preferred_max_chunk_ms must be positive")
    split: list[_Interval] = []
    for interval in intervals:
        pending = [interval]
        while pending:
            current = pending.pop(0)
            duration = current.end_ms - current.start_ms
            if duration <= preferred_max_ms:
                split.append(current)
                continue
            split_at = _best_context_split_ms(current, frames, threshold, config, preferred_max_ms, hard_max_ms)
            if split_at is None:
                if not config.soft_split_allowed:
                    split.append(_Interval(current.start_ms, current.end_ms, current.split_reason, _add_risk_flag(current.risk_flags, "segment_over_hard_max")))
                    continue
                split_at = min(current.start_ms + hard_max_ms, current.end_ms)
                left_reason = "vad_hard_split"
                left_flags = _add_risk_flag(current.risk_flags, "hard_split")
            else:
                left_reason = "vad_silence_split"
                left_flags = current.risk_flags
            left = _Interval(current.start_ms, split_at, left_reason, left_flags)
            right = _Interval(split_at, current.end_ms, current.split_reason, current.risk_flags)
            split.append(left)
            if right.end_ms > right.start_ms:
                pending.insert(0, right)
    return split


def _best_context_split_ms(
    interval: _Interval,
    frames: list[_Frame],
    threshold: float,
    config: VadConfig,
    preferred_max_ms: int,
    hard_max_ms: int,
) -> int | None:
    earliest = interval.start_ms + max(1, min(config.target_min_chunk_ms, preferred_max_ms))
    preferred = interval.start_ms + preferred_max_ms
    latest = min(interval.start_ms + hard_max_ms, interval.end_ms)
    if earliest >= latest:
        return None
    gaps = _silence_gaps(frames, threshold, interval.start_ms, interval.end_ms)
    candidates = [gap for gap in gaps if gap[1] >= earliest and gap[0] <= preferred and gap[0] < latest]
    if not candidates:
        candidates = [gap for gap in gaps if gap[1] >= earliest and gap[0] < latest]
    if not candidates:
        return None
    candidates.sort(key=lambda gap: (abs(gap[1] - preferred), -(gap[1] - gap[0]), gap[1]))
    split_at = candidates[0][1]
    if split_at <= interval.start_ms or split_at >= interval.end_ms:
        return None
    return split_at


def _silence_gaps(frames: list[_Frame], threshold: float, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for frame in frames:
        if frame.end_ms <= start_ms or frame.start_ms >= end_ms:
            continue
        frame_start = max(start_ms, frame.start_ms)
        frame_end = min(end_ms, frame.end_ms)
        if frame.rms < threshold:
            if current_start is None:
                current_start = frame_start
            current_end = frame_end
        elif current_start is not None and current_end is not None:
            gaps.append((current_start, current_end))
            current_start = None
            current_end = None
    if current_start is not None and current_end is not None:
        gaps.append((current_start, current_end))
    return gaps


def _merge_risk_flags(first: list[str], second: list[str]) -> list[str]:
    merged = list(first)
    for flag in second:
        if flag not in merged:
            merged.append(flag)
    return merged


def _add_risk_flag(flags: list[str], flag: str) -> list[str]:
    return flags if flag in flags else flags + [flag]


def _split_long_intervals(
    intervals: list[_Interval],
    max_duration_ms: int,
    soft_split_allowed: bool,
) -> list[_Interval]:
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be positive")
    split: list[_Interval] = []
    for interval in intervals:
        duration = interval.end_ms - interval.start_ms
        if duration <= max_duration_ms:
            split.append(interval)
            continue
        if not soft_split_allowed:
            split.append(_Interval(interval.start_ms, interval.end_ms, interval.split_reason, ["segment_over_max_duration"]))
            continue
        start = interval.start_ms
        while start < interval.end_ms:
            end = min(start + max_duration_ms, interval.end_ms)
            split.append(_Interval(start, end, "vad_soft_split", interval.risk_flags))
            start = end
    return split
