from __future__ import annotations

from dataclasses import dataclass

from dubber.asr.timestamps import TimestampUnit
from dubber.core.models import ASRChunkingConfig


@dataclass(frozen=True)
class WordTimestampChunk:
    start_ms: int
    end_ms: int
    word_start_ms: int
    word_end_ms: int
    split_threshold_ms: int
    trailing_silence_ms: int
    risk_flags: list[str]
    units: list[TimestampUnit]


def build_word_timestamp_chunks(
    words: list[TimestampUnit],
    *,
    audio_duration_ms: int,
    config: ASRChunkingConfig,
) -> list[WordTimestampChunk]:
    if not words:
        return []
    ordered = sorted(words, key=lambda unit: (unit.start_ms, unit.end_ms))
    groups, threshold = _split_with_silence_threshold(ordered, config)
    chunks: list[WordTimestampChunk] = []
    for index, (group, hard_split) in enumerate(groups):
        next_group = groups[index + 1][0] if index + 1 < len(groups) else None
        chunks.append(
            _build_chunk(
                group,
                next_word_start_ms=next_group[0].start_ms if next_group else None,
                audio_duration_ms=audio_duration_ms,
                split_threshold_ms=threshold,
                trailing_silence_cap_ms=config.trailing_silence_cap_ms,
                hard_split=hard_split,
            )
        )
    return chunks


def _split_with_silence_threshold(
    words: list[TimestampUnit],
    config: ASRChunkingConfig,
) -> tuple[list[tuple[list[TimestampUnit], bool]], int]:
    threshold = config.initial_silence_ms
    while True:
        groups = _split_at_gaps(words, threshold)
        if all(_word_span_duration(group) <= config.max_chunk_duration_ms for group in groups):
            return [(group, False) for group in groups], threshold
        if threshold <= config.min_silence_ms:
            return _hard_split_groups(groups, config.max_chunk_duration_ms), threshold
        threshold = max(config.min_silence_ms, threshold - config.silence_step_ms)


def _split_at_gaps(words: list[TimestampUnit], threshold_ms: int) -> list[list[TimestampUnit]]:
    groups: list[list[TimestampUnit]] = []
    current: list[TimestampUnit] = [words[0]]
    for unit in words[1:]:
        previous = current[-1]
        if max(0, unit.start_ms - previous.end_ms) >= threshold_ms:
            groups.append(current)
            current = [unit]
        else:
            current.append(unit)
    groups.append(current)
    return groups


def _hard_split_groups(
    groups: list[list[TimestampUnit]],
    max_duration_ms: int,
) -> list[tuple[list[TimestampUnit], bool]]:
    output: list[tuple[list[TimestampUnit], bool]] = []
    for group in groups:
        if _word_span_duration(group) <= max_duration_ms:
            output.append((group, False))
            continue
        current: list[TimestampUnit] = []
        for unit in group:
            if current and unit.end_ms - current[0].start_ms > max_duration_ms:
                output.append((current, True))
                current = [unit]
            else:
                current.append(unit)
        if current:
            output.append((current, True))
    return output


def _build_chunk(
    units: list[TimestampUnit],
    *,
    next_word_start_ms: int | None,
    audio_duration_ms: int,
    split_threshold_ms: int,
    trailing_silence_cap_ms: int,
    hard_split: bool,
) -> WordTimestampChunk:
    word_start_ms = units[0].start_ms
    word_end_ms = units[-1].end_ms
    max_end_ms = max(word_end_ms, audio_duration_ms)
    end_ms = min(max_end_ms, word_end_ms + trailing_silence_cap_ms)
    if next_word_start_ms is not None:
        end_ms = min(end_ms, next_word_start_ms)
    end_ms = max(word_end_ms, end_ms)
    risk_flags = ["hard_split"] if hard_split else []
    return WordTimestampChunk(
        start_ms=word_start_ms,
        end_ms=end_ms,
        word_start_ms=word_start_ms,
        word_end_ms=word_end_ms,
        split_threshold_ms=split_threshold_ms,
        trailing_silence_ms=max(0, end_ms - word_end_ms),
        risk_flags=risk_flags,
        units=list(units),
    )


def _word_span_duration(units: list[TimestampUnit]) -> int:
    if not units:
        return 0
    return units[-1].end_ms - units[0].start_ms
