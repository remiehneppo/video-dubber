# Video Dubber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-first Vietnamese commentary dubbing pipeline with resumable file-based jobs.

**Architecture:** Start with a modular monolith and a strict artifact contract. The first milestone creates the core package, config loader, workspace path manager, manifest, checkpoint store, and CLI skeleton before media or provider code.

**Tech Stack:** Python 3.12, pytest, stdlib JSON/YAML fallback parsing, FFmpeg/ffprobe for later media stages.

---

## Implementation Checklist

### Phase 0: Core Foundation

- [x] Create Python project scaffold.
- [x] Add config example and loader.
- [x] Add core stage/status models.
- [x] Add workspace path manager with path traversal protection.
- [x] Add atomic JSON write helper.
- [x] Add artifact manifest with SHA-256 validation.
- [x] Add checkpoint/job state store.
- [x] Add JSON logger.
- [x] Add CLI skeleton with `run`, `resume`, `status`, `validate`, and `jobs`.
- [x] Add unit tests for config, paths, manifest, and checkpoint behavior.

### Phase 1: One-Minute Vertical Slice

- [x] Implement FFmpeg adapter for metadata, audio extraction, simple muxing.
- [x] Implement Stage 0 job init.
- [x] Implement Stage 1 audio extraction with metadata artifact.
- [x] Implement WAV energy VAD with merge/split behavior.
- [x] Implement mock ASR provider.
- [x] Implement mock glossary/translation provider.
- [x] Implement mock TTS provider that generates timed WAV tone/silence.
- [x] Implement simple mix/render stage.
- [x] Add integration test for input video to output mp4.

### Phase 2: Glossary and Translation

- [x] Add glossary extraction prompt and artifact schema.
- [x] Add `waiting_review` pause.
- [x] Add resume after `glossary.locked.json`.
- [x] Add translation block builder.
- [x] Add translation validator and glossary consistency checks.
- [x] Add compression pass hook.

### Phase 3: TTS Alignment and Audio QA

- [x] Add duration planner.
- [x] Add TTS manifest.
- [x] Add segment-level ASR/TTS checkpoint.
- [x] Add time-stretch/overflow policy.
- [x] Add ducking envelope and loudness normalization.
- [x] Add QA report.

### Phase 4: Resume Hardening

- [x] Add artifact corruption detection.
- [x] Add rerun stage command.
- [x] Add rerun segment command.
- [x] Add crash injection tests for TTS resume.

### Phase 5: Web Monitor

- [x] Add FastAPI app.
- [x] Add job status endpoint.
- [x] Add progress websocket.
- [x] Add output preview/download endpoints.

### Phase 6: Productionization

- [x] Add OpenAI-compatible ASR provider adapter.
- [x] Add OpenAI-compatible LLM provider adapter.
- [x] Add OpenAI-compatible TTS provider adapter.
- [x] Add provider config sections and env placeholders.
- [x] Add provider mode validation for OpenAI-compatible configuration.
- [x] Wire OpenAI-compatible provider mode through ASR, glossary, translation, and TTS pipeline stages.
- [x] Add README.
- [x] Add runbook.
- [x] Add package console script and build metadata.
- [x] Add Dockerfile.
- [x] Add Docker Compose.
- [x] Add CI workflow.

## Task 1: Core Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `config.example.yaml`
- Create: `dubber/__init__.py`
- Create: `dubber/core/config.py`
- Create: `dubber/core/enums.py`
- Create: `dubber/core/io.py`
- Create: `dubber/core/models.py`
- Create: `dubber/core/paths.py`
- Create: `dubber/orchestrator/artifact_manifest.py`
- Create: `dubber/orchestrator/checkpoint_store.py`
- Create: `dubber/observability/logger.py`
- Test: `tests/unit/test_core_foundation.py`

- [x] **Step 1: Write failing tests**

Run: `pytest tests/unit/test_core_foundation.py -q`

Expected: FAIL because `dubber` modules do not exist yet.

- [x] **Step 2: Implement minimal core modules**

Implement only the behavior required by the tests:
- default config loading from YAML-like example file,
- workspace directory creation,
- safe relative artifact paths,
- atomic JSON writes,
- SHA-256 manifest entries,
- job state persistence.

- [x] **Step 3: Verify tests pass**

Run: `pytest tests/unit/test_core_foundation.py -q`

Expected: PASS.

## Task 2: CLI Skeleton

**Files:**
- Create: `cli.py`
- Test: `tests/unit/test_cli.py`

- [x] **Step 1: Write failing CLI tests**

Run: `pytest tests/unit/test_cli.py -q`

Expected: FAIL because `cli.py` does not exist yet.

- [x] **Step 2: Implement minimal CLI**

Implement `jobs`, `status`, and `validate` against the file workspace. Leave `run` and `resume` present but conservative until Stage 0 exists.

- [x] **Step 3: Verify CLI tests pass**

Run: `pytest tests/unit/test_cli.py -q`

Expected: PASS.

## Self-Review

- Scope is intentionally limited to Phase 0 and the first CLI shell before media stages.
- No provider/API secrets are required.
- No web UI is included in the first implementation batch.
- The plan avoids placeholder commands in the active tasks.
