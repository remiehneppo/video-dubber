from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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

