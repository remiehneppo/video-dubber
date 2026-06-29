from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranslationContextBlock:
    target_segments: list[Mapping[str, Any]]
    context_before: list[Mapping[str, Any]]
    context_after: list[Mapping[str, Any]]

    @property
    def all_segments(self) -> list[Mapping[str, Any]]:
        return [*self.context_before, *self.target_segments, *self.context_after]


def build_translation_blocks(
    segments: list[Mapping[str, Any]],
    *,
    block_size: int,
    overlap: int,
) -> list[list[Mapping[str, Any]]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= block_size:
        raise ValueError("overlap must be smaller than block_size")
    if not segments:
        return []

    blocks: list[list[Mapping[str, Any]]] = []
    step = block_size - overlap
    start = 0
    while start < len(segments):
        block = segments[start : start + block_size]
        blocks.append(block)
        if start + block_size >= len(segments):
            break
        start += step
    return blocks


def build_translation_context_blocks(
    segments: list[Mapping[str, Any]],
    *,
    min_context_words: int,
    max_context_words: int,
    context_overlap_words: int,
    target_segment_count: int,
) -> list[TranslationContextBlock]:
    if min_context_words < 1:
        raise ValueError("min_context_words must be positive")
    if max_context_words < min_context_words:
        raise ValueError("max_context_words must be at least min_context_words")
    if context_overlap_words < 0:
        raise ValueError("context_overlap_words must be non-negative")
    if target_segment_count < 1:
        raise ValueError("target_segment_count must be positive")
    if not segments:
        return []

    blocks: list[TranslationContextBlock] = []
    start = 0
    while start < len(segments):
        end = _target_end_index(
            segments,
            start,
            min_context_words=min_context_words,
            max_context_words=max_context_words,
            target_segment_count=target_segment_count,
        )
        blocks.append(
            TranslationContextBlock(
                target_segments=segments[start:end],
                context_before=_collect_context_before(segments, start, context_overlap_words),
                context_after=_collect_context_after(segments, end, context_overlap_words),
            )
        )
        start = end
    return blocks


def _target_end_index(
    segments: list[Mapping[str, Any]],
    start: int,
    *,
    min_context_words: int,
    max_context_words: int,
    target_segment_count: int,
) -> int:
    end = start
    words = 0
    while end < len(segments) and end - start < target_segment_count:
        next_words = _segment_word_count(segments[end])
        if end > start and words + next_words > max_context_words:
            break
        words += next_words
        end += 1
        if words >= min_context_words:
            break
    return max(start + 1, end)


def _collect_context_before(
    segments: list[Mapping[str, Any]],
    start: int,
    word_target: int,
) -> list[Mapping[str, Any]]:
    if word_target <= 0 or start <= 0:
        return []
    collected: list[Mapping[str, Any]] = []
    words = 0
    index = start - 1
    while index >= 0 and words < word_target:
        collected.insert(0, segments[index])
        words += _segment_word_count(segments[index])
        index -= 1
    return collected


def _collect_context_after(
    segments: list[Mapping[str, Any]],
    end: int,
    word_target: int,
) -> list[Mapping[str, Any]]:
    if word_target <= 0 or end >= len(segments):
        return []
    collected: list[Mapping[str, Any]] = []
    words = 0
    index = end
    while index < len(segments) and words < word_target:
        collected.append(segments[index])
        words += _segment_word_count(segments[index])
        index += 1
    return collected


def _segment_word_count(segment: Mapping[str, Any]) -> int:
    text = str(segment.get("source_text", ""))
    return len(text.split())
