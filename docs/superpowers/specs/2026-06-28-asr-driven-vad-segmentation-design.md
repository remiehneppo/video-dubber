# ASR-Driven VAD Segmentation Design

## Goal

Optimize VAD so it preserves complete spoken context by producing longer ASR chunks, then derive final transcript segments from ASR timestamps.

## Decision

Use option B: prefer word timestamps and fall back to ASR segment timestamps.

VAD is no longer the final source of dubbing segment timing. VAD creates ASR chunks that are large enough for local faster-whisper to understand context. ASR returns timestamps. A transcript segmentation module normalizes timestamps back to the original video timeline and creates final transcript segments for glossary, translation, TTS, and mixing.

## Domain Terms

- **ASR chunk**: A longer audio interval sent to ASR. It is optimized for recognition context and GPU efficiency, not final dubbing timing.
- **Transcript segment**: A final timeline interval with source text. It is derived from ASR timestamps and is the segment used by glossary, translation, TTS, and mixing.

## Requirements

- VAD should create ASR chunks with target duration 20-45 seconds.
- VAD may extend chunks up to a hard maximum when there is no good pause to split.
- VAD should not drop short speech islands when they can be merged into a neighboring chunk.
- ASR requests should ask for timestamp output.
- Word timestamps are preferred when the provider returns them.
- Segment timestamps are accepted as fallback.
- If timestamps are required and neither word nor segment timestamps exist, ASR should fail clearly.
- Local faster-whisper/Speaches should avoid provider-side VAD filtering during chunk transcription because the pipeline already controls chunking.
- Final `transcript.v1.json` remains the downstream artifact read by glossary, translation, and TTS.
- Final transcript segments must carry `timestamp_source`, `timestamp_quality`, and `risk_flags`.

## Config Shape

```yaml
vad:
  mode: asr_context_chunks
  frame_ms: 100
  threshold_ratio: 0.08
  min_speech_duration_ms: 700
  target_min_chunk_ms: 20000
  preferred_max_chunk_ms: 45000
  hard_max_chunk_ms: 90000
  silence_merge_threshold_ms: 2500
  context_padding_ms: 1500
  soft_split_allowed: true

asr_service:
  timestamp_mode: prefer_word
  require_timestamps: true
  allow_chunk_text_fallback: false
  vad_filter: false

transcript_segmentation:
  target_min_segment_ms: 8000
  preferred_max_segment_ms: 25000
  max_segment_ms: 45000
  min_pause_split_ms: 600
  prefer_punctuation_split: true
```

## Pipeline

1. `run_vad` publishes VAD intervals as ASR chunks in `segments.v1.json`. The artifact name stays stable initially to limit downstream churn, but its metadata will identify `kind: asr_chunks`.
2. `run_asr` extracts each ASR chunk, requests timestamps, writes raw ASR JSON, normalizes timestamps, and builds final transcript segments.
3. `transcript.v1.json` contains final transcript segments, not one row per VAD chunk.
4. Glossary and translation continue reading `transcript.v1.json`.
5. TTS should iterate over final transcript segments instead of VAD chunks so generated speech aligns with transcript-derived timing.

## Risks

- Provider schemas vary. The implementation must normalize both OpenAI-style `words` and faster-whisper-style `segments`.
- Word timestamps cost more GPU. This is accepted.
- Segment fallback is less accurate than word timestamps, so the artifact must expose timestamp source and quality.
- Changing TTS to use transcript segments can affect checkpoint IDs. Segment IDs should remain stable and deterministic.

## Verification

- Unit tests cover timestamp request payloads.
- Unit tests cover ASR timestamp normalization for word and segment formats.
- Unit tests cover transcript segmentation by pause, punctuation, minimum duration, and maximum duration.
- Pipeline tests verify `transcript.v1.json` can contain more or fewer rows than VAD chunks and TTS uses transcript segments.
- A real local ASR smoke test should inspect the provider response schema before full video runs.
