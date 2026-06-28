# ASR-Driven VAD Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ASR-driven segmentation where VAD creates context-rich chunks and final transcript segments are derived from ASR timestamps.

**Architecture:** Keep VAD as the ASR chunk generator. Add a transcript segmentation module that normalizes ASR word or segment timestamps onto the original timeline and produces final transcript segments for glossary, translation, and TTS. Keep `transcript.v1.json` as the downstream seam so glossary/translation change minimally.

**Tech Stack:** Python dataclasses, pytest, httpx OpenAI-compatible provider, existing file-based job artifacts.

---

## File Structure

- Modify `CONTEXT.md`: add `ASR chunk` and `Transcript segment` domain terms.
- Modify `dubber/core/models.py`: extend `VadConfig`, `ASRServiceConfig`, and add `TranscriptSegmentationConfig`.
- Modify `dubber/core/config.py`: load new config fields with backward-compatible defaults.
- Modify `config.example.yaml` and `config.local.yaml`: add chunk/timestamp/segmentation config.
- Modify `dubber/providers/base.py`: extend `ASRResult` with optional normalized timestamp metadata if needed.
- Modify `dubber/providers/asr_openai_compatible.py`: request `response_format=verbose_json`, word timestamps when configured, and `vad_filter=false`.
- Create `dubber/asr/timestamps.py`: normalize provider raw response into timestamp units.
- Create `dubber/transcript/segmentation.py`: build final transcript segments from normalized ASR timestamps.
- Modify `dubber/audio/vad.py`: support context chunk config names and hard chunk max semantics.
- Modify `dubber/pipeline/stages.py`: build transcript from ASR timestamps and make TTS iterate transcript segments.
- Add tests in `tests/unit/test_openai_providers.py`, `tests/unit/test_transcript_segmentation.py`, `tests/unit/test_pipeline_asr_segments.py`, `tests/unit/test_pipeline_vad_config.py`, and `tests/unit/test_tts_segment_producer.py` as needed.

## Task 1: Domain and Config Shape

**Files:**
- Modify: `CONTEXT.md`
- Modify: `dubber/core/models.py`
- Modify: `dubber/core/config.py`
- Modify: `config.example.yaml`
- Modify: `config.local.yaml`
- Test: `tests/unit/test_core_foundation.py`

- [ ] **Step 1: Write failing config test**

Add a test that loads a config with `vad.mode`, ASR timestamp fields, and transcript segmentation fields, then asserts the dataclass values.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/unit/test_core_foundation.py`

Expected: fail because new config fields do not exist.

- [ ] **Step 3: Implement config dataclasses and loader**

Add backward-compatible defaults:

```python
mode: str = "asr_context_chunks"
min_speech_duration_ms: int = 700
target_min_chunk_ms: int = 20_000
preferred_max_chunk_ms: int = 45_000
hard_max_chunk_ms: int = 90_000
timestamp_mode: str = "prefer_word"
require_timestamps: bool = True
allow_chunk_text_fallback: bool = False
vad_filter: bool = False
```

- [ ] **Step 4: Run config tests**

Run: `pytest -q tests/unit/test_core_foundation.py tests/unit/test_pipeline_vad_config.py`

Expected: pass.

## Task 2: ASR Timestamp Request and Normalization

**Files:**
- Modify: `dubber/providers/asr_openai_compatible.py`
- Create: `dubber/asr/timestamps.py`
- Modify: `dubber/providers/base.py`
- Test: `tests/unit/test_openai_providers.py`

- [ ] **Step 1: Write failing provider payload test**

Assert the ASR provider sends:

```python
data={
    "model": model,
    "language": language,
    "response_format": "verbose_json",
    "timestamp_granularities[]": "word",
    "vad_filter": "false",
}
```

- [ ] **Step 2: Write failing normalization tests**

Cover these raw shapes:

```python
{"words": [{"word": "Hello", "start": 1.0, "end": 1.4}], "text": "Hello"}
{"segments": [{"text": "Hello.", "start": 1.0, "end": 2.0}], "text": "Hello."}
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest -q tests/unit/test_openai_providers.py`

Expected: fail because timestamp request and normalizer do not exist.

- [ ] **Step 4: Implement provider request fields and normalizer**

Normalizer returns timestamp source `word`, `segment`, or raises when timestamps are required and missing.

- [ ] **Step 5: Run ASR provider tests**

Run: `pytest -q tests/unit/test_openai_providers.py`

Expected: pass.

## Task 3: Transcript Segmentation Module

**Files:**
- Create: `dubber/transcript/segmentation.py`
- Test: `tests/unit/test_transcript_segmentation.py`

- [ ] **Step 1: Write failing tests**

Test that word timestamps split on punctuation and pauses, merge too-short segments, and hard split at `max_segment_ms`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest -q tests/unit/test_transcript_segmentation.py`

Expected: fail because module does not exist.

- [ ] **Step 3: Implement minimal segmentation**

Create final segment dictionaries with:

```python
segment_id
start_ms
end_ms
source_text
confidence
timestamp_source
timestamp_quality
risk_flags
raw_response_path
source_chunk_id
```

- [ ] **Step 4: Run transcript tests**

Run: `pytest -q tests/unit/test_transcript_segmentation.py`

Expected: pass.

## Task 4: VAD Context Chunking

**Files:**
- Modify: `dubber/audio/vad.py`
- Test: `tests/unit/test_vad.py`
- Test: `tests/unit/test_pipeline_vad_config.py`

- [ ] **Step 1: Write failing VAD tests**

Add tests that short speech islands are merged into context chunks and long intervals prefer 20-45 second chunking before hard max.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest -q tests/unit/test_vad.py tests/unit/test_pipeline_vad_config.py`

Expected: fail on new chunk behavior.

- [ ] **Step 3: Implement chunk semantics**

Map old config names to new behavior where needed, keep existing tests passing, and mark segment artifact metadata as `kind: asr_chunks`.

- [ ] **Step 4: Run VAD tests**

Run: `pytest -q tests/unit/test_vad.py tests/unit/test_pipeline_vad_config.py`

Expected: pass.

## Task 5: Pipeline Integration

**Files:**
- Modify: `dubber/pipeline/stages.py`
- Test: `tests/unit/test_pipeline_asr_segments.py`
- Test: `tests/unit/test_tts_segment_producer.py`

- [ ] **Step 1: Write failing ASR pipeline test**

Make a fake ASR provider return word timestamps for one chunk and assert `transcript.v1.json` contains timestamp-derived final transcript segments.

- [ ] **Step 2: Write failing TTS pipeline test**

Assert TTS iterates `transcript.v1.json` segments, not raw VAD chunks.

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest -q tests/unit/test_pipeline_asr_segments.py tests/unit/test_tts_segment_producer.py`

Expected: fail before integration.

- [ ] **Step 4: Integrate timestamp segmentation in `run_asr`**

Use raw ASR response, source chunk timing, and segmentation config to produce final transcript segments.

- [ ] **Step 5: Update TTS to use transcript segments**

Keep translations joined by final transcript `segment_id`.

- [ ] **Step 6: Run integration-adjacent tests**

Run: `pytest -q tests/unit/test_pipeline_asr_segments.py tests/unit/test_tts_segment_producer.py tests/unit/test_translation.py`

Expected: pass.

## Task 6: Verification

**Files:**
- No new files unless a test reveals a focused fix.

- [ ] **Step 1: Run focused suite**

Run:

```bash
pytest -q tests/unit/test_openai_providers.py tests/unit/test_transcript_segmentation.py tests/unit/test_vad.py tests/unit/test_pipeline_vad_config.py tests/unit/test_pipeline_asr_segments.py tests/unit/test_tts_segment_producer.py tests/unit/test_translation.py
```

Expected: pass.

- [ ] **Step 2: Inspect git diff**

Run: `git diff --stat` and `git diff --check`.

Expected: no whitespace errors; changed files match this plan.

- [ ] **Step 3: Optional live ASR schema smoke test**

Run a small local transcription request against `http://127.0.0.1:23334/v1/audio/transcriptions` and inspect whether response has `words` or `segments`.

Expected: response has at least one timestamp source.

## Self-Review

- Spec coverage: config, ASR timestamp request, timestamp fallback, transcript segmentation, VAD chunking, pipeline/TTS integration, diagnostics, and verification are covered.
- Placeholder scan: no task depends on an undefined later concept; each task has files and commands.
- Type consistency: task names use `ASR chunk`, `Transcript segment`, `timestamp_source`, and `risk_flags` consistently with the design spec.
