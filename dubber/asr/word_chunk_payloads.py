from __future__ import annotations

from typing import Any

from dubber.asr.timestamps import NormalizedASRTimestamps, TimestampUnit
from dubber.asr.word_chunking import build_word_timestamp_chunks
from dubber.core.models import ASRChunkingConfig


def build_asr_word_chunk_payloads(
    asr_chunks: list[dict[str, Any]],
    *,
    audio_duration_ms: int,
    config: ASRChunkingConfig,
) -> dict[str, list[dict[str, Any]]]:
    records = _word_records(asr_chunks)
    word_chunks = build_word_timestamp_chunks(
        [record["unit"] for record in records],
        audio_duration_ms=audio_duration_ms,
        config=config,
    )
    artifact_chunks: list[dict[str, Any]] = []
    transcript_chunks: list[dict[str, Any]] = []
    cursor = 0
    for index, word_chunk in enumerate(word_chunks, start=1):
        chunk_id = f"wchunk_{index:06d}"
        chunk_records = records[cursor : cursor + len(word_chunk.units)]
        cursor += len(word_chunk.units)
        source_ids = _dedupe([str(record["source_chunk_id"]) for record in chunk_records])
        raw_paths = _dedupe([str(record["raw_response_path"]) for record in chunk_records])
        timestamp_risk_flags = _dedupe(
            [
                *word_chunk.risk_flags,
                *[str(flag) for record in chunk_records for flag in record["timestamp_risk_flags"]],
            ]
        )
        vad_risk_flags = _dedupe(
            [str(flag) for record in chunk_records for flag in record["vad_risk_flags"]]
        )
        normalized = NormalizedASRTimestamps(
            source="word",
            quality="word_chunk",
            units=word_chunk.units,
            risk_flags=timestamp_risk_flags,
        )
        artifact_chunks.append(
            {
                "chunk_id": chunk_id,
                "start_ms": word_chunk.start_ms,
                "end_ms": word_chunk.end_ms,
                "word_start_ms": word_chunk.word_start_ms,
                "word_end_ms": word_chunk.word_end_ms,
                "split_threshold_ms": word_chunk.split_threshold_ms,
                "trailing_silence_ms": word_chunk.trailing_silence_ms,
                "risk_flags": _dedupe([*timestamp_risk_flags, *vad_risk_flags]),
                "source_bootstrap_segment_ids": source_ids,
                "raw_response_paths": raw_paths,
                "words": [_timestamp_unit_to_dict(unit) for unit in word_chunk.units],
            }
        )
        transcript_chunks.append(
            {
                "chunk_id": chunk_id,
                "raw_response_path": raw_paths[0] if raw_paths else "",
                "timestamps": normalized,
                "confidence": min(float(record["confidence"]) for record in chunk_records) if chunk_records else 1.0,
                "vad_split_reason": ",".join(
                    _dedupe([str(record["vad_split_reason"]) for record in chunk_records if record["vad_split_reason"]])
                ),
                "vad_risk_flags": vad_risk_flags,
                "silence_before_ms": int(chunk_records[0]["silence_before_ms"]) if chunk_records else 0,
                "silence_after_ms": word_chunk.trailing_silence_ms,
                "source_bootstrap_segment_ids": source_ids,
            }
        )
    return {"chunks": artifact_chunks, "asr_chunks": transcript_chunks}


def _word_records(asr_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in asr_chunks:
        timestamps = chunk["timestamps"]
        if not isinstance(timestamps, NormalizedASRTimestamps):
            raise TypeError("chunk timestamps must be NormalizedASRTimestamps")
        for unit in timestamps.units:
            records.append(
                {
                    "unit": unit,
                    "source_chunk_id": str(chunk.get("chunk_id", "")),
                    "raw_response_path": str(chunk.get("raw_response_path", "")),
                    "confidence": float(chunk.get("confidence", 1.0)),
                    "timestamp_risk_flags": list(timestamps.risk_flags),
                    "vad_split_reason": str(chunk.get("vad_split_reason", "")),
                    "vad_risk_flags": list(chunk.get("vad_risk_flags", [])),
                    "silence_before_ms": int(chunk.get("silence_before_ms", 0)),
                }
            )
    return sorted(records, key=lambda record: (record["unit"].start_ms, record["unit"].end_ms))


def _timestamp_unit_to_dict(unit: TimestampUnit) -> dict[str, object]:
    return {"text": unit.text.strip(), "start_ms": unit.start_ms, "end_ms": unit.end_ms}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
