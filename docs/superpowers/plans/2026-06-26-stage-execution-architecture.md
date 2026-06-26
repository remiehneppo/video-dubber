# Stage Execution Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the job stage execution and artifact publication modules so `JobManager` orchestrates job lifecycle while stage implementation details live behind smaller seams.

**Architecture:** Keep the modular monolith and existing file-based job workspace. Introduce a `StageContext` module to hold shared stage dependencies and a `StageArtifacts` publication module to centralize JSON writes, manifest recording, and checkpoint updates. Move stage bodies out of `JobManager` incrementally, preserving CLI, resume, rerun, provider mode, and artifact file behavior.

**Tech Stack:** Python 3.11, dataclasses, pytest, existing FFmpeg adapter, existing file-based `WorkspacePaths`, `CheckpointStore`, `SegmentCheckpointStore`, and `ArtifactManifest`.

---

## Scope

This plan implements the first architecture-deepening slice only:

- Stage execution module
- Artifact publication module
- TTS segment production module enough to remove duplication between full-run TTS and `rerun-segment`

This plan does not redesign provider prompts, Web monitor read models, or Commentary Mode QA. Those should remain separate plans.

## File Structure

- Create: `dubber/pipeline/stage_context.py`
  - Holds `StageContext`, the shared interface each stage needs: workspace paths, stores, manifest, ffmpeg adapter, provider mode, provider bundle, and helper methods for reading artifacts.
- Create: `dubber/orchestrator/stage_artifacts.py`
  - Deepens artifact publication: write JSON artifact, record manifest entry, save manifest, and mark stage progress in one place.
- Create: `dubber/pipeline/stages.py`
  - Contains stage functions for job init, audio extract, VAD, ASR, glossary, translation, TTS, and mixing.
- Create: `dubber/tts/segment_producer.py`
  - Owns “produce one aligned TTS segment” for both full TTS runs and segment reruns.
- Modify: `dubber/pipeline/job_manager.py`
  - Keep public interface: `run`, `resume`, `rerun_stage`, `rerun_segment`.
  - Replace extracted stage method bodies with calls into `dubber.pipeline.stages`.
- Test: `tests/unit/test_stage_artifacts.py`
  - Tests artifact publication ordering and manifest/checkpoint updates.
- Test: `tests/unit/test_tts_segment_producer.py`
  - Tests one-segment TTS production and checkpoint-compatible output shape.
- Modify existing tests:
  - `tests/integration/test_mock_vertical_slice.py`
  - `tests/integration/test_openai_provider_mode.py`

---

### Task 1: Add StageContext

**Files:**
- Create: `dubber/pipeline/stage_context.py`
- Test through existing import checks in later tasks.

- [ ] **Step 1: Create the context module**

Add this file:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubber.core.io import read_json
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.providers.factory import ProviderBundle
from dubber.providers.ffmpeg import FFmpegAdapter


@dataclass
class StageContext:
    paths: WorkspacePaths
    store: CheckpointStore
    manifest: ArtifactManifest
    ffmpeg: FFmpegAdapter
    provider_mode: str = "mock"
    provider_bundle: ProviderBundle | None = None

    def artifact_json(self, filename: str) -> dict[str, Any]:
        return read_json(self.paths.artifact_path(filename))

    def resolve_input(self) -> Path:
        return self.paths.resolve_relative(self.store.state.input_file)

    def require_provider_bundle(self) -> ProviderBundle:
        if self.provider_bundle is None:
            raise RuntimeError("provider bundle is not configured")
        return self.provider_bundle
```

- [ ] **Step 2: Run a smoke import**

Run:

```bash
python3 -c "from dubber.pipeline.stage_context import StageContext; print(StageContext.__name__)"
```

Expected:

```text
StageContext
```

- [ ] **Step 3: Commit**

```bash
git add dubber/pipeline/stage_context.py
git commit -m "refactor: add stage context"
```

---

### Task 2: Add Artifact Publication Module

**Files:**
- Create: `dubber/orchestrator/stage_artifacts.py`
- Test: `tests/unit/test_stage_artifacts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_stage_artifacts.py`:

```python
from __future__ import annotations

from pathlib import Path

from dubber.core.enums import StageName, StageStatus
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts


def test_publish_json_records_manifest_and_stage(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_test", input_file=Path("input/source.mp4"))
    manifest = ArtifactManifest.create("job_test", paths.manifest_file)
    publisher = StageArtifacts(paths=paths, store=store, manifest=manifest)

    artifact_path = publisher.publish_json(
        stage=StageName.VAD,
        name="segments",
        filename="segments.v1.json",
        payload={"schema_version": "1.0", "segments": []},
        status=StageStatus.COMPLETED,
        done=0,
        total=0,
    )

    assert artifact_path == paths.artifact_path("segments.v1.json")
    assert artifact_path.exists()
    assert manifest.validate_artifact("segments", 1)
    assert store.state.stages[StageName.VAD].status == StageStatus.COMPLETED
    assert store.state.stages[StageName.VAD].artifact == "artifacts/segments.v1.json"
    assert store.state.stages[StageName.VAD].done == 0
    assert store.state.stages[StageName.VAD].total == 0


def test_publish_existing_file_records_manifest_and_stage(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_test", input_file=Path("input/source.mp4"))
    manifest = ArtifactManifest.create("job_test", paths.manifest_file)
    publisher = StageArtifacts(paths=paths, store=store, manifest=manifest)
    final_audio = paths.audio_dir / "final_mix.wav"
    final_audio.write_bytes(b"fake audio")

    publisher.publish_file(
        stage=StageName.MIXING,
        name="final_audio",
        path=final_audio,
        status=StageStatus.RUNNING,
    )

    assert manifest.validate_artifact("final_audio", 1)
    assert store.state.stages[StageName.MIXING].status == StageStatus.RUNNING
    assert store.state.stages[StageName.MIXING].artifact == "audio/final_mix.wav"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/unit/test_stage_artifacts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dubber.orchestrator.stage_artifacts'`.

- [ ] **Step 3: Implement StageArtifacts**

Create `dubber/orchestrator/stage_artifacts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore


@dataclass
class StageArtifacts:
    paths: WorkspacePaths
    store: CheckpointStore
    manifest: ArtifactManifest

    def publish_json(
        self,
        *,
        stage: StageName,
        name: str,
        filename: str,
        payload: dict[str, Any],
        version: int = 1,
        schema_version: str = "1.0",
        status: StageStatus = StageStatus.COMPLETED,
        done: int | None = None,
        total: int | None = None,
    ) -> Path:
        path = self.paths.artifact_path(filename)
        write_json_atomic(path, payload)
        self.publish_file(
            stage=stage,
            name=name,
            path=path,
            version=version,
            schema_version=schema_version,
            status=status,
            done=done,
            total=total,
        )
        return path

    def publish_file(
        self,
        *,
        stage: StageName,
        name: str,
        path: Path,
        version: int = 1,
        schema_version: str = "1.0",
        status: StageStatus = StageStatus.COMPLETED,
        done: int | None = None,
        total: int | None = None,
    ) -> None:
        self.manifest.record_artifact(
            name=name,
            version=version,
            path=path,
            created_by_stage=stage,
            schema_version=schema_version,
        )
        self.manifest.save()
        self.store.mark_stage(
            stage,
            status,
            artifact=self.paths.to_relative(path),
            done=done,
            total=total,
        )
        self.store.save()
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/unit/test_stage_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dubber/orchestrator/stage_artifacts.py tests/unit/test_stage_artifacts.py
git commit -m "refactor: add stage artifact publisher"
```

---

### Task 3: Extract Job Init, Audio Extract, and VAD Stages

**Files:**
- Create: `dubber/pipeline/stages.py`
- Modify: `dubber/pipeline/job_manager.py`
- Test: existing `tests/integration/test_mock_vertical_slice.py`

- [ ] **Step 1: Create stage functions for the first three stages**

Create `dubber/pipeline/stages.py` with:

```python
from __future__ import annotations

import shutil
from pathlib import Path

from dubber.audio.vad import VadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext


def run_job_init(ctx: StageContext, copied_input: Path) -> None:
    ctx.store.mark_stage(StageName.JOB_INIT, StageStatus.RUNNING)
    ctx.store.save()
    metadata_path = ctx.paths.input_dir / "input_metadata.v1.json"
    write_json_atomic(metadata_path, ctx.ffmpeg.probe(copied_input))
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
        stage=StageName.JOB_INIT,
        name="input_metadata",
        path=metadata_path,
    )


def run_audio_extract(ctx: StageContext, copied_input: Path) -> int:
    ctx.store.mark_stage(StageName.AUDIO_EXTRACT, StageStatus.RUNNING)
    ctx.store.save()
    original_wav = ctx.paths.audio_dir / "original.wav"
    ctx.ffmpeg.extract_audio(copied_input, original_wav)
    shutil.copy2(original_wav, ctx.paths.audio_dir / "vocals.wav")
    duration_ms = ctx.ffmpeg.duration_ms(copied_input)
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.AUDIO_EXTRACT,
        name="audio_analysis",
        filename="audio_analysis.v1.json",
        payload={
            "schema_version": "1.0",
            "audio_duration_ms": duration_ms,
            "sample_rate": 44100,
            "channels": 1,
            "source_separation_used": False,
            "source_separation_reason": "mock_vertical_slice",
        },
    )
    return duration_ms


def run_vad(ctx: StageContext) -> None:
    ctx.store.mark_stage(StageName.VAD, StageStatus.RUNNING)
    ctx.store.save()
    segments = detect_segments(
        ctx.paths.audio_dir / "vocals.wav",
        VadConfig(
            min_duration_ms=300,
            max_duration_ms=25_000,
            silence_merge_threshold_ms=400,
        ),
    )
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.VAD,
        name="segments",
        filename="segments.v1.json",
        payload={
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "source_audio": "audio/vocals.wav",
            "segments": [segment.to_dict() for segment in segments],
        },
    )
```

- [ ] **Step 2: Wire `JobManager.run` to use StageContext and extracted functions**

In `dubber/pipeline/job_manager.py`, add imports:

```python
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import run_audio_extract, run_job_init, run_vad
```

Inside `run`, after `manifest = ArtifactManifest.create(...)`, create:

```python
ctx = StageContext(
    paths=paths,
    store=store,
    manifest=manifest,
    ffmpeg=self.ffmpeg,
    provider_mode=self.provider_mode,
    provider_bundle=self.provider_bundle,
)
```

Replace:

```python
self._stage_job_init(copied_input, paths, store, manifest)
duration_ms = self._stage_audio_extract(copied_input, paths, store, manifest)
self._stage_vad(paths, store, manifest)
```

with:

```python
run_job_init(ctx, copied_input)
duration_ms = run_audio_extract(ctx, copied_input)
run_vad(ctx)
```

Keep the old private methods temporarily until all stage functions are extracted.

- [ ] **Step 3: Run vertical slice**

Run:

```bash
pytest tests/integration/test_mock_vertical_slice.py::test_run_mock_vertical_slice_creates_output_video -q
```

Expected: PASS.

- [ ] **Step 4: Run artifact publisher tests**

Run:

```bash
pytest tests/unit/test_stage_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dubber/pipeline/stage_context.py dubber/pipeline/stages.py dubber/pipeline/job_manager.py
git commit -m "refactor: extract initial stage execution"
```

---

### Task 4: Extract ASR, Glossary, and Translation Stages

**Files:**
- Modify: `dubber/pipeline/stages.py`
- Modify: `dubber/pipeline/job_manager.py`
- Test: `tests/integration/test_openai_provider_mode.py`
- Test: `tests/integration/test_mock_vertical_slice.py`

- [ ] **Step 1: Move ASR, glossary, and translation logic into stage functions**

Append these functions to `dubber/pipeline/stages.py`:

```python
import asyncio

from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations


def run_asr(ctx: StageContext) -> None:
    segments = ctx.artifact_json("segments.v1.json")["segments"]
    ctx.store.mark_stage(StageName.ASR, StageStatus.RUNNING, done=0, total=len(segments))
    ctx.store.save()
    segment_store = SegmentCheckpointStore.create(
        ctx.paths.artifact_path("asr_segments.v1.json"),
        stage=StageName.ASR.value,
        segment_ids=[str(segment["segment_id"]) for segment in segments],
    )
    transcript_segments = []
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment["segment_id"])
        raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
        if ctx.provider_mode == "openai_compatible":
            asr_result = asyncio.run(ctx.require_provider_bundle().asr.transcribe(ctx.paths.audio_dir / "vocals.wav", language="en"))
            text = asr_result.text
            confidence = asr_result.confidence if asr_result.confidence is not None else 1.0
            write_json_atomic(raw_path, asr_result.raw)
        else:
            text = f"Mock transcript segment {index} for vertical slice."
            confidence = 1.0
            write_json_atomic(raw_path, {"text": text, "confidence": confidence})
        segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
        segment_store.save()
        transcript_segments.append(
            {
                "segment_id": segment_id,
                "start_ms": int(segment["start_ms"]),
                "end_ms": int(segment["end_ms"]),
                "source_text": text,
                "confidence": confidence,
                "asr_warnings": [],
                "raw_response_path": ctx.paths.to_relative(raw_path),
            }
        )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(stage=StageName.ASR, name="asr_segments", path=segment_store.path, status=StageStatus.RUNNING)
    publisher.publish_json(
        stage=StageName.ASR,
        name="transcript",
        filename="transcript.v1.json",
        payload={
            "schema_version": "1.0",
            "provider": {"type": ctx.provider_mode, "model": "provider-asr" if ctx.provider_mode == "openai_compatible" else "mock-asr"},
            "segments": transcript_segments,
        },
        done=len(segments),
        total=len(segments),
    )


def run_glossary(ctx: StageContext, *, glossary_review: bool) -> bool:
    ctx.store.mark_stage(StageName.GLOSSARY, StageStatus.RUNNING)
    ctx.store.save()
    glossary_path = ctx.paths.artifact_path("glossary.draft.json" if glossary_review else "glossary.locked.json")
    source_segments = [segment["segment_id"] for segment in ctx.artifact_json("segments.v1.json")["segments"]]
    if ctx.provider_mode == "openai_compatible":
        transcript = ctx.artifact_json("transcript.v1.json")
        glossary_result = asyncio.run(ctx.require_provider_bundle().llm.complete_json("extract glossary", "terminology " + str(transcript["segments"]), schema={"type": "object"}))
        terms = [
            {
                "term_id": term.get("term_id", f"term_{index:04d}"),
                "original": term.get("original", ""),
                "vietnamese": term.get("vietnamese", ""),
                "category": term.get("category", "term"),
                "confidence": term.get("confidence", 1.0),
                "locked": not glossary_review,
                "source_segments": term.get("source_segments", source_segments),
                "notes": term.get("notes", ""),
            }
            for index, term in enumerate(glossary_result.get("terms", []), start=1)
        ]
    else:
        terms = [{
            "term_id": "term_0001",
            "original": "vertical slice",
            "vietnamese": "lát cắt dọc",
            "category": "phrase",
            "locked": not glossary_review,
            "source_segments": source_segments,
            "notes": "Mock glossary entry.",
        }]
    payload = {"schema_version": "1.0", "domain": ctx.provider_mode, "status": "draft" if glossary_review else "locked", "terms": terms}
    write_json_atomic(glossary_path, payload)
    if glossary_review:
        StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
            stage=StageName.GLOSSARY,
            name="glossary_draft",
            path=glossary_path,
            status=StageStatus.WAITING_REVIEW,
        )
        return True
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
        stage=StageName.GLOSSARY,
        name="glossary",
        path=glossary_path,
    )
    return False


def run_translation(ctx: StageContext) -> None:
    transcript = ctx.artifact_json("transcript.v1.json")
    ctx.store.mark_stage(StageName.TRANSLATION, StageStatus.RUNNING, done=0, total=len(transcript["segments"]))
    ctx.store.save()
    glossary = ctx.artifact_json("glossary.locked.json")
    translated_segments = []
    provider_translations: dict[str, dict] = {}
    if ctx.provider_mode == "openai_compatible":
        translation_result = asyncio.run(
            ctx.require_provider_bundle().llm.complete_json(
                "translate transcript to Vietnamese",
                str({"segments": transcript["segments"], "glossary": glossary["terms"]}),
                schema={"type": "object"},
            )
        )
        provider_translations = {
            str(segment.get("segment_id")): segment
            for segment in translation_result.get("segments", [])
            if segment.get("segment_id") is not None
        }
    for source in transcript["segments"]:
        if ctx.provider_mode == "openai_compatible":
            provider_segment = provider_translations.get(str(source["segment_id"]), {})
            candidate = {
                "segment_id": source["segment_id"],
                "source_text": source["source_text"],
                "vi_text": provider_segment.get("vi_text", ""),
                "used_terms": provider_segment.get("used_terms", []),
                "length_ratio": provider_segment.get("length_ratio", 1.0),
                "translation_warnings": provider_segment.get("translation_warnings", []),
            }
        else:
            candidate = {
                "segment_id": source["segment_id"],
                "source_text": source["source_text"],
                "vi_text": f"Đây là lát cắt dọc thuyết minh tiếng Việt mẫu cho {source['segment_id']}.",
                "used_terms": ["vertical slice"],
                "length_ratio": 1.0,
                "translation_warnings": [],
            }
        compressed = compress_segment_translation(candidate, glossary["terms"], max_length_ratio=3.0)
        candidate["vi_text"] = compressed.vi_text
        candidate["translation_warnings"] = candidate["translation_warnings"] + compressed.warnings
        translated_segments.append(candidate)
    validation = validate_translations(transcript["segments"], translated_segments, glossary["terms"], max_length_ratio=3.0)
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.TRANSLATION,
        name="translated",
        filename="translated.v1.json",
        payload={
            "schema_version": "1.0",
            "segments": translated_segments,
            "validation_warnings": validation.warnings,
        },
        done=len(transcript["segments"]),
        total=len(transcript["segments"]),
    )
```

- [ ] **Step 2: Wire `JobManager.run`, `resume`, and `rerun_stage`**

Import:

```python
from dubber.pipeline.stages import run_asr, run_glossary, run_translation
```

Replace calls:

```python
self._stage_asr(paths, store, manifest)
if self._stage_glossary(paths, store, manifest, glossary_review=options.glossary_review):
self._stage_translation(paths, store, manifest)
```

with:

```python
run_asr(ctx)
if run_glossary(ctx, glossary_review=options.glossary_review):
run_translation(ctx)
```

For `resume` and `rerun_stage`, create a `StageContext` from loaded `paths`, `store`, and `manifest`, then call `run_translation(ctx)` instead of `_stage_translation(...)`.

- [ ] **Step 3: Run provider-mode integration**

Run:

```bash
pytest tests/integration/test_openai_provider_mode.py -q
```

Expected: PASS.

- [ ] **Step 4: Run glossary resume and translation rerun tests**

Run:

```bash
pytest tests/integration/test_mock_vertical_slice.py::test_glossary_review_pauses_then_resume_creates_output_video tests/integration/test_mock_vertical_slice.py::test_rerun_translation_repairs_corrupt_translated_artifact -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dubber/pipeline/stages.py dubber/pipeline/job_manager.py
git commit -m "refactor: extract language stages"
```

---

### Task 5: Add TTS Segment Producer

**Files:**
- Create: `dubber/tts/segment_producer.py`
- Test: `tests/unit/test_tts_segment_producer.py`

- [ ] **Step 1: Write failing unit test**

Create `tests/unit/test_tts_segment_producer.py`:

```python
from __future__ import annotations

import wave
from pathlib import Path

from dubber.core.paths import WorkspacePaths
from dubber.tts.segment_producer import produce_mock_tts_segment


def test_produce_mock_tts_segment_returns_manifest_row(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    segment = {
        "segment_id": "seg_000001",
        "start_ms": 1000,
        "end_ms": 2000,
        "duration_ms": 1000,
    }

    row = produce_mock_tts_segment(paths=paths, segment=segment)

    assert row["segment_id"] == "seg_000001"
    assert row["orig_duration_ms"] == 1000
    assert row["tts_duration_ms"] == 1100
    assert row["alignment_action"] == "time_stretch"
    assert row["raw_audio_path"] == "tts/seg_000001.raw.wav"
    assert row["audio_path"] == "tts/seg_000001.wav"
    assert (paths.root / row["audio_path"]).exists()
    with wave.open(str(paths.root / row["audio_path"]), "rb") as wav:
        assert wav.getnchannels() == 1
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/unit/test_tts_segment_producer.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dubber.tts.segment_producer'`.

- [ ] **Step 3: Implement mock segment producer**

Create `dubber/tts/segment_producer.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from dubber.core.paths import WorkspacePaths
from dubber.providers.factory import ProviderBundle
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.aligner import apply_time_stretch
from dubber.tts.duration_planner import plan_segment_duration
from dubber.tts.mock import synthesize_tone_wav


def produce_mock_tts_segment(*, paths: WorkspacePaths, segment: dict[str, Any]) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    orig_ms = int(segment["duration_ms"])
    raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
    tts_duration_ms = max(100, int(orig_ms * 1.1))
    synthesize_tone_wav(raw_audio_path, tts_duration_ms)
    return _align_segment(
        paths=paths,
        segment=segment,
        raw_audio_path=raw_audio_path,
        tts_duration_ms=tts_duration_ms,
        provider_metadata={},
    )


async def produce_provider_tts_segment(
    *,
    paths: WorkspacePaths,
    segment: dict[str, Any],
    text: str,
    provider_bundle: ProviderBundle,
    ffmpeg: FFmpegAdapter,
    voice: str = "default",
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
    tts_result = await provider_bundle.tts.synthesize(text, voice=voice, output_path=raw_audio_path)
    tts_duration_ms = tts_result.duration_ms or ffmpeg.duration_ms(tts_result.audio_path)
    return _align_segment(
        paths=paths,
        segment=segment,
        raw_audio_path=raw_audio_path,
        tts_duration_ms=tts_duration_ms,
        provider_metadata=tts_result.provider_metadata,
    )


def _align_segment(
    *,
    paths: WorkspacePaths,
    segment: dict[str, Any],
    raw_audio_path: Path,
    tts_duration_ms: int,
    provider_metadata: dict[str, Any],
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    orig_ms = int(segment["duration_ms"])
    aligned_audio_path = paths.tts_dir / f"{segment_id}.wav"
    timing_plan = plan_segment_duration(
        segment_id,
        orig_duration_ms=orig_ms,
        tts_duration_ms=tts_duration_ms,
    )
    apply_time_stretch(raw_audio_path, aligned_audio_path, timing_plan.stretch_ratio if timing_plan.action == "time_stretch" else 1.0)
    return {
        "segment_id": segment["segment_id"],
        "target_start_ms": int(segment["start_ms"]) + 500,
        "target_end_ms": int(segment["end_ms"]),
        "original_start_ms": int(segment["start_ms"]),
        "original_end_ms": int(segment["end_ms"]),
        "commentary_delay_ms": 500,
        "orig_duration_ms": orig_ms,
        "tts_duration_ms": tts_duration_ms,
        "alignment_action": timing_plan.action,
        "stretch_ratio": timing_plan.stretch_ratio,
        "overflow_ms": timing_plan.overflow_ms,
        "raw_audio_path": paths.to_relative(raw_audio_path),
        "audio_path": paths.to_relative(aligned_audio_path),
        "warnings": timing_plan.warnings,
        "provider_metadata": provider_metadata,
    }
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
pytest tests/unit/test_tts_segment_producer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dubber/tts/segment_producer.py tests/unit/test_tts_segment_producer.py
git commit -m "refactor: add tts segment producer"
```

---

### Task 6: Extract TTS and Mixing Stages

**Files:**
- Modify: `dubber/pipeline/stages.py`
- Modify: `dubber/pipeline/job_manager.py`
- Test: `tests/integration/test_mock_vertical_slice.py`
- Test: `tests/integration/test_openai_provider_mode.py`

- [ ] **Step 1: Add TTS and mixing stage functions**

Append to `dubber/pipeline/stages.py`:

```python
from dubber.tts.mock import synthesize_tone_wav
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_segment


def run_tts(
    ctx: StageContext,
    duration_ms: int,
    *,
    crash_stage: str | None = None,
    crash_after_segments: int | None = None,
) -> Path:
    segments = ctx.artifact_json("segments.v1.json")["segments"]
    ctx.store.mark_stage(StageName.TTS, StageStatus.RUNNING, done=0, total=len(segments))
    ctx.store.save()
    segment_store = SegmentCheckpointStore.create(
        ctx.paths.artifact_path("tts_segments.v1.json"),
        stage=StageName.TTS.value,
        segment_ids=[str(segment["segment_id"]) for segment in segments],
    )
    if crash_stage == StageName.TTS.value and crash_after_segments == 0:
        segment_store.save()
        raise RuntimeError("simulated crash at tts after 0 segments")
    translations_by_id = {
        str(segment["segment_id"]): segment
        for segment in ctx.artifact_json("translated.v1.json")["segments"]
    }
    mix_audio_path = ctx.paths.tts_dir / "mix.wav"
    synthesize_tone_wav(mix_audio_path, duration_ms)
    tts_segments = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        if ctx.provider_mode == "openai_compatible":
            row = asyncio.run(
                produce_provider_tts_segment(
                    paths=ctx.paths,
                    segment=segment,
                    text=translations_by_id.get(segment_id, {}).get("vi_text", ""),
                    provider_bundle=ctx.require_provider_bundle(),
                    ffmpeg=ctx.ffmpeg,
                )
            )
        else:
            row = produce_mock_tts_segment(paths=ctx.paths, segment=segment)
        tts_segments.append(row)
        segment_store.mark(segment_id, StageStatus.COMPLETED, artifact=row["audio_path"])
        segment_store.save()
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(stage=StageName.TTS, name="tts_segments", path=segment_store.path, status=StageStatus.RUNNING)
    publisher.publish_json(
        stage=StageName.TTS,
        name="tts_manifest",
        filename="tts_manifest.v1.json",
        payload={"schema_version": "1.0", "segments": tts_segments},
        done=len(segments),
        total=len(segments),
    )
    return mix_audio_path


def run_mixing(ctx: StageContext, copied_input: Path, tts_audio: Path, duration_ms: int) -> Path:
    ctx.store.mark_stage(StageName.MIXING, StageStatus.RUNNING)
    ctx.store.save()
    final_audio = ctx.paths.audio_dir / "final_mix.wav"
    ctx.ffmpeg.mix_commentary_audio(ctx.paths.audio_dir / "original.wav", tts_audio, final_audio)
    output_video = ctx.paths.output_dir / f"{copied_input.stem}_vi.mp4"
    ctx.ffmpeg.mux_video_audio(copied_input, final_audio, output_video)
    qa_path = ctx.paths.output_dir / f"{copied_input.stem}_vi.qa.json"
    write_json_atomic(
        qa_path,
        {
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "input_duration_ms": duration_ms,
            "output_duration_ms": ctx.ffmpeg.duration_ms(output_video),
            "segments_total": len(ctx.artifact_json("segments.v1.json")["segments"]),
            "low_confidence_segments": 0,
            "glossary_terms": 1,
            "tts_overflow_segments": 0,
            "max_overflow_ms": 0,
            "sync_drift_p95_ms": 0,
            "warnings": [],
        },
    )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(stage=StageName.MIXING, name="final_audio", path=final_audio, status=StageStatus.RUNNING)
    publisher.publish_file(stage=StageName.MIXING, name="qa_report", path=qa_path, status=StageStatus.RUNNING)
    publisher.publish_file(stage=StageName.MIXING, name="output_video", path=output_video, status=StageStatus.COMPLETED)
    return output_video
```

- [ ] **Step 2: Wire `JobManager` full run, resume, and rerun paths**

Import:

```python
from dubber.pipeline.stages import run_mixing, run_tts
```

Replace `_stage_tts(...)` with `run_tts(ctx, ...)`.

Replace `_stage_mixing(...)` with `run_mixing(ctx, ...)`.

Create `StageContext` in `resume` and `rerun_stage` before stage calls:

```python
ctx = StageContext(
    paths=paths,
    store=store,
    manifest=manifest,
    ffmpeg=self.ffmpeg,
    provider_mode=self.provider_mode,
    provider_bundle=self.provider_bundle,
)
```

- [ ] **Step 3: Run crash/resume and provider tests**

Run:

```bash
pytest tests/integration/test_mock_vertical_slice.py::test_resume_recovers_from_tts_crash_injection tests/integration/test_openai_provider_mode.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full mock integration file**

Run:

```bash
pytest tests/integration/test_mock_vertical_slice.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dubber/pipeline/stages.py dubber/pipeline/job_manager.py dubber/tts/segment_producer.py
git commit -m "refactor: extract tts and mixing stages"
```

---

### Task 7: Replace `rerun_segment` TTS Duplication

**Files:**
- Modify: `dubber/pipeline/job_manager.py`
- Test: `tests/integration/test_mock_vertical_slice.py`

- [ ] **Step 1: Update `rerun_segment` to use `produce_mock_tts_segment`**

In `dubber/pipeline/job_manager.py`, import:

```python
from dubber.tts.segment_producer import produce_mock_tts_segment
```

Inside `rerun_segment`, replace manual raw WAV synthesis, duration planning, and time stretch with:

```python
row = produce_mock_tts_segment(paths=paths, segment=segment)
audio_path = paths.resolve_relative(row["audio_path"])
checkpoint.mark(segment_id, StageStatus.COMPLETED, artifact=row["audio_path"])
checkpoint.save()

tts_manifest_path = paths.artifact_path("tts_manifest.v1.json")
tts_manifest = read_json(tts_manifest_path)
for item in tts_manifest["segments"]:
    if str(item["segment_id"]) == segment_id:
        item.update(row)
write_json_atomic(tts_manifest_path, tts_manifest)
```

Keep the existing manifest records, mixing call, and job completion logic.

- [ ] **Step 2: Run segment rerun test**

Run:

```bash
pytest tests/integration/test_mock_vertical_slice.py::test_rerun_tts_segment_repairs_segment_checkpoint_and_output -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add dubber/pipeline/job_manager.py
git commit -m "refactor: reuse tts segment producer for rerun"
```

---

### Task 8: Remove Extracted Private Stage Methods

**Files:**
- Modify: `dubber/pipeline/job_manager.py`
- Test: full suite

- [ ] **Step 1: Remove unused imports from `job_manager.py`**

After all stage calls are extracted, remove unused imports:

```python
import asyncio
from dubber.audio.vad import VadConfig, detect_segments
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.tts.aligner import apply_time_stretch
from dubber.tts.duration_planner import plan_segment_duration
from dubber.tts.mock import synthesize_tone_wav
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations
```

Keep imports still needed by `rerun_segment`, `validate`, and lifecycle code.

- [ ] **Step 2: Delete old private methods**

Delete these methods from `JobManager`:

```python
_stage_job_init
_stage_audio_extract
_stage_vad
_stage_asr
_stage_glossary
_stage_translation
_stage_tts
_stage_mixing
```

Do not delete:

```python
_validate_input
_write_resolved_config
```

- [ ] **Step 3: Run syntax check**

Run:

```bash
python3 -m compileall cli.py dubber web tests
```

Expected: no compile errors.

- [ ] **Step 4: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add dubber/pipeline/job_manager.py
git commit -m "refactor: slim job manager stage methods"
```

---

### Task 9: Document the New Module Responsibilities

**Files:**
- Modify: `README.md`
- Optional create: `CONTEXT.md`

- [ ] **Step 1: Add architecture notes to README**

Add under the first paragraph in `README.md`:

```markdown
## Architecture Notes

- `JobManager` owns job lifecycle: create workspace, start, resume, rerun, and mark final job status.
- `StageContext` carries stage dependencies without making each stage know how jobs are constructed.
- Stage functions in `dubber/pipeline/stages.py` own stage behavior and artifact shapes.
- `StageArtifacts` owns artifact publication: atomic write, manifest record, manifest save, checkpoint update.
- TTS segment production lives in `dubber/tts/segment_producer.py` so full TTS runs and segment reruns share alignment behavior.
```

- [ ] **Step 2: Create domain context file**

Create `CONTEXT.md`:

```markdown
# Video Dubber Domain Context

## Core Terms

- **Job workspace**: File-based directory for one video dubbing job. Contains state, manifest, copied input, audio, raw provider responses, artifacts, TTS audio, output, and logs.
- **Artifact manifest**: Source of truth for produced files, checksums, stage ownership, and schema versions.
- **Checkpoint**: Persisted job or segment progress used for resume and rerun.
- **Stage**: Ordered job step that reads prior artifacts and publishes one or more new artifacts.
- **Commentary Mode**: Output mode where Vietnamese speech overlays ducked original speech while background audio remains.
- **Glossary review**: Human pause where extracted terms are edited and locked before translation.
- **Provider mode**: Runtime selection between mock local behavior and OpenAI-compatible ASR, LLM, and TTS adapters.
```

- [ ] **Step 3: Run docs asset tests**

Run:

```bash
pytest tests/unit/test_docs_assets.py tests/unit/test_assets.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md CONTEXT.md
git commit -m "docs: record stage architecture terms"
```

---

### Task 10: Final Verification

**Files:**
- No new code expected.

- [ ] **Step 1: Check git diff**

Run:

```bash
git diff --stat HEAD~9..HEAD
```

Expected: changes are limited to stage architecture files, tests, README, and `CONTEXT.md`.

- [ ] **Step 2: Run full tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run CLI smoke command**

Run:

```bash
python3 cli.py --help
```

Expected: command list includes `run`, `resume`, `status`, `jobs`, `validate`, `rerun`, `rerun-segment`, and `web`.

- [ ] **Step 4: Commit any verification-only doc correction**

If Task 10 revealed only documentation corrections, commit them:

```bash
git add README.md CONTEXT.md docs/superpowers/plans/2026-06-26-stage-execution-architecture.md
git commit -m "docs: finalize stage architecture plan"
```

If no files changed, do not create an empty commit.

---

## Self-Review

**Spec coverage:** This plan covers the selected architecture candidates: stage execution, artifact publication, and TTS segment production. It intentionally leaves provider language workflow deepening, Web monitor read models, and Commentary Mode QA as future plans.

**Placeholder scan:** No `TBD`, `TODO`, “similar to,” or unspecified test steps are used.

**Type consistency:** `StageContext`, `StageArtifacts`, `run_*` stage function names, and `produce_*_tts_segment` names are introduced before use and used consistently across tasks.
