# Video Dubber

CLI-first Vietnamese commentary dubbing pipeline for educational videos, lectures, podcasts, and explainer content.

The current implementation is a modular monolith with file-based jobs, manifest-tracked artifacts, checkpoint/resume, a mock provider mode for local end-to-end runs, OpenAI-compatible provider adapters, and a FastAPI monitor.

## Architecture Notes

- `JobManager` owns job lifecycle: create workspace, start, resume, rerun, and mark final job status.
- `StageContext` carries stage dependencies without making each stage know how jobs are constructed.
- Stage functions in `dubber/pipeline/stages.py` own stage behavior and artifact shapes.
- `StageArtifacts` owns artifact publication: atomic write, manifest record, manifest save, checkpoint update.
- TTS segment production lives in `dubber/tts/segment_producer.py` so full TTS runs and segment reruns share alignment behavior.

## Requirements

- Python 3.11+
- FFmpeg and ffprobe on PATH
- Python dependencies from `pyproject.toml`

Install locally:

```bash
python3 -m pip install -e .
```

## Configure Providers

Copy `.env.example` and set provider values for real ASR, LLM, and TTS services. The mock vertical slice does not require API keys.

```bash
cp .env.example .env
```

`config.example.yaml` supports these provider sections:

- `asr_service`
- `llm_service`
- `tts_service`

## Provider Mode Status

`--provider-mode mock` is the supported local end-to-end mode and does not require API keys. `--provider-mode openai_compatible` builds the configured ASR, LLM, and TTS adapters and runs the same pipeline against those services when valid provider settings and credentials are supplied.

## Run A Mock End-To-End Job

```bash
dubber run --input samples/short_1min.mp4 --provider-mode mock --no-glossary-review
# or: python3 cli.py run --input samples/short_1min.mp4 --provider-mode mock --no-glossary-review
```

This creates a workspace under `workspace/{job_id}` and emits an output MP4 under `output/` inside that job workspace.

## Glossary Review Flow

```bash
python3 cli.py run --input lecture.mp4 --provider-mode mock --glossary-review
```

The job pauses with `waiting_review` and writes `workspace/<job_id>/artifacts/glossary.draft.json`.
Review the terms, then save `workspace/<job_id>/artifacts/glossary.locked.json`.
The locked file has the same glossary schema as the draft, but must use `status: "locked"`.
For accepted terms, set `locked: true`; keep protected terms protected.

Minimal shape:

```json
{
  "schema_version": "1.0",
  "domain": "mathematics",
  "domain_profile": "calculus",
  "status": "locked",
  "terms": [
    {
      "term_id": "protected_dr",
      "original": "dr",
      "vietnamese": "d r",
      "category": "calculus_notation",
      "confidence": 1.0,
      "locked": true,
      "protected": true,
      "spoken": "d r",
      "display": "dr",
      "forbidden": ["doctor", "bác sĩ", "tiến sĩ"],
      "source_segments": ["seg_000001"]
    }
  ]
}
```

To accept a glossary draft mechanically, run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
job = Path("workspace/<job_id>")
draft = json.loads((job / "artifacts/glossary.draft.json").read_text(encoding="utf-8"))
draft["status"] = "locked"
for term in draft.get("terms", []):
    term["locked"] = True
(job / "artifacts/glossary.locked.json").write_text(
    json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

Then resume:

```bash
dubber resume --job <job_id>
# or: python3 cli.py resume --job <job_id>
```

### High-Risk Cue Review Flow

After translation, the job may pause with `waiting_review` and write
`workspace/<job_id>/artifacts/review.required.json`. This is separate from glossary
review. It means a cue needs human approval before TTS because source normalization,
protected spans, or ASR/timeline risk flags could affect spoken audio.

Common reasons:

- `source_normalization_review_required`: verify `source_text_normalized`, especially technical notation and protected spans.
- `asr_timeline_review_required`: verify cue text/timing when ASR word timestamps were repaired or VAD produced over-long speech segments.

Create `workspace/<job_id>/artifacts/review.locked.json` with this shape:

```json
{
  "schema_version": "1.0",
  "status": "locked",
  "cues": [
    {
      "cue_id": "cue_abc123",
      "review_overrides": {
        "source_text_normalized": "original source text after approved normalization",
        "display_text": "Vietnamese subtitle text",
        "spoken_text": "Vietnamese text for TTS",
        "protected_spans": []
      }
    }
  ]
}
```

To accept all current review suggestions mechanically, run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
job = Path("workspace/<job_id>")
required = json.loads((job / "artifacts/review.required.json").read_text(encoding="utf-8"))
locked = {
    "schema_version": "1.0",
    "status": "locked",
    "cues": [
        {
            "cue_id": cue["cue_id"],
            "review_overrides": cue.get("review_overrides", {}),
        }
        for cue in required.get("cues", [])
    ],
}
(job / "artifacts/review.locked.json").write_text(
    json.dumps(locked, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

Resume normally after locking review:

```bash
python3 cli.py resume --job <job_id> --workspace workspace
```

Important review rules:

- `display_text` is for subtitles; `spoken_text` is for TTS. Keep both intentional.
- Preserve `protected_spans`; for technical notation, `spoken` and `forbidden` matter for validation and TTS normalization.
- Only edit `start_ms`/`end_ms` when you are deliberately applying a safe timing override; keep intervals non-overlapping.
- If a job already finished ASR/translation, prefer `resume` over a fresh `run` to avoid repeating provider calls.
- For non-calculus videos, pass `--domain-profile generic` or a suitable profile so calculus protected-span rules do not dominate prompts.

## Status And Validation

```bash
python3 cli.py jobs
python3 cli.py status --job <job_id>
python3 cli.py validate --job <job_id>
```

## Rerun Commands

```bash
python3 cli.py rerun --job <job_id> --stage translation
python3 cli.py rerun-segment --job <job_id> --stage tts --segment seg_000001
```

## Batch Processing

Process every supported video directly inside one directory (non-recursive):

```bash
dubber batch run --input-dir ./videos --workspace workspace --provider-mode mock --no-glossary-review
```

A batch is stored at `workspace/batch_<id>/` with `batch_state.json`,
`batch_manifest.json`, shared glossary files under `artifacts/`, and independent
job workspaces under `jobs/`. Inputs are sorted by filename and copied into each
job, so later resume operations do not depend on the source directory.

`runtime.max_parallel_jobs` controls the number of videos processed at once and
defaults to `1`. Each worker creates its own job manager and provider context.
One failed video does not stop the remaining videos; the final batch status is
`partial_failed` when completed and failed jobs coexist.

For a reviewed shared glossary:

```bash
dubber batch run --input-dir ./videos --glossary-review
# Review artifacts/glossary.draft.json and save artifacts/glossary.locked.json
dubber batch resume --batch batch_<id>
```

Inspect or validate a batch with:

```bash
dubber batch status --batch batch_<id>
dubber batch validate --batch batch_<id>
```

Use repeated `--job` options to resume selected jobs. With no `--job`, resume
selects every unfinished job. `--from-stage` forces invalidation of that stage
and all downstream job artifacts; without `--job` it applies to the whole
batch. Re-running ASR or an earlier stage never rebuilds the already locked
batch glossary.

## Checkpoints And Resume

`dubber resume --job <id>` finds the earliest incomplete stage or the first
stage whose manifest checksum is invalid. A completed, valid job is a no-op.
Use `--from-stage asr` (or another stage) to explicitly reset that stage and all
downstream artifacts while retaining valid upstream work.

ASR and TTS persist per-segment checkpoints. Glossary and translation persist a
checkpoint and response artifact after every provider block. Resume loads these
files, discards completed units whose referenced artifact is missing, and only
calls providers for the remaining units.

## Web Monitor

```bash
dubber web --workspace workspace --host 127.0.0.1 --port 8080
# or: python3 cli.py web --workspace workspace --host 127.0.0.1 --port 8080
```

API endpoints:

- `GET /api/batches`
- `GET /api/batches/{batch_id}`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/qa`
- `GET /api/jobs/{job_id}/output`
- `WS /ws/jobs/{job_id}`

## Test

```bash
pytest -q
```
