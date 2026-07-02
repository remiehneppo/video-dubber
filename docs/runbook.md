# Video Dubber Runbook

## Glossary review

Symptom: job status is `waiting_review` and current stage is `glossary`.

Action:

1. Open `workspace/{job_id}/artifacts/glossary.draft.json`.
2. Review `terms[].original`, `terms[].vietnamese`, `spoken`, `display`, `forbidden`, and `protected`.
3. Save the reviewed file as `workspace/{job_id}/artifacts/glossary.locked.json` with `status: "locked"`.
4. Set `locked: true` on accepted terms. Do not remove protected terms unless you are intentionally disabling that domain guard.
5. Run `python3 cli.py resume --job {job_id}`.

Accept draft mechanically:

```bash
python3 - <<'PY'
import json
from pathlib import Path
job = Path("workspace/{job_id}")
draft = json.loads((job / "artifacts/glossary.draft.json").read_text(encoding="utf-8"))
draft["status"] = "locked"
for term in draft.get("terms", []):
    term["locked"] = True
(job / "artifacts/glossary.locked.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
```

For batch-level glossary review, use `workspace/{batch_id}/artifacts/glossary.draft.json` and save `workspace/{batch_id}/artifacts/glossary.locked.json`, then run `python3 cli.py batch resume --batch {batch_id}`.

## High-risk cue review

Symptom: job status is `waiting_review` after translation, and `workspace/{job_id}/artifacts/review.required.json` exists.

Action:

1. Run `python3 cli.py review --workspace workspace --job {job_id}`.
2. Inspect each cue's `reason`, `risk_flags`, `source_text_raw`, `source_text_normalized`, `display_text`, `spoken_text`, and `protected_spans`.
3. Accept or edit each cue interactively.
4. The command writes `workspace/{job_id}/artifacts/review.locked.json`.
5. Run `python3 cli.py resume --job {job_id}`.

`review.locked.json` shape:

```json
{
  "schema_version": "1.0",
  "status": "locked",
  "cues": [
    {
      "cue_id": "cue_abc123",
      "review_overrides": {
        "source_text_normalized": "approved source text",
        "display_text": "subtitle text",
        "spoken_text": "tts text",
        "protected_spans": []
      }
    }
  ]
}
```

Accept all suggested review overrides mechanically:

```bash
python3 - <<'PY'
import json
from pathlib import Path
job = Path("workspace/{job_id}")
required = json.loads((job / "artifacts/review.required.json").read_text(encoding="utf-8"))
locked = {
    "schema_version": "1.0",
    "status": "locked",
    "cues": [
        {"cue_id": cue["cue_id"], "review_overrides": cue.get("review_overrides", {})}
        for cue in required.get("cues", [])
    ],
}
(job / "artifacts/review.locked.json").write_text(json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
```

Review notes:

- `source_normalization_review_required`: check technical terms and any deterministic/LLM source-normalization suggestions.
- `asr_timeline_review_required`: check text and timing if ASR timestamps were repaired or VAD produced over-long speech segments.
- `display_text` feeds subtitles; `spoken_text` feeds TTS. They may differ intentionally.
- Preserve protected spans and forbidden translations for technical notation.
- Timing overrides (`start_ms`, `end_ms`) should be rare and must not overlap adjacent cues.
- Prefer `resume` after creating lock files. Use `--from-stage` only when you intentionally invalidate that stage and downstream artifacts.

## Crash during TTS

Symptom: job status is `failed`, current stage is `tts`, and `last_error` contains a provider or simulated crash message.

Action:

1. Inspect `workspace/{job_id}/artifacts/tts_segments.v1.json`.
2. Confirm completed segments remain marked `completed`.
3. Run `python3 cli.py resume --job {job_id}` to recover the TTS stage and render output.

## Corrupt artifact

Symptom: `python3 cli.py validate --job {job_id}` reports an invalid artifact such as `translated.v2`.

Action:

1. Identify the corrupt stage from the artifact name.
2. Rerun the stage, for example `python3 cli.py rerun --job {job_id} --stage translation`.
3. Run `python3 cli.py validate --job {job_id}` again.

## Segment-level TTS repair

Symptom: a single TTS segment failed or has bad audio.

Action:

1. Inspect `workspace/{job_id}/artifacts/tts_segments.v1.json`.
2. Run `python3 cli.py rerun-segment --job {job_id} --stage tts --segment seg_000001`.
3. Validate the job and preview the output.

## Web monitor

Start the local monitor:

```bash
python3 cli.py web --workspace workspace --host 127.0.0.1 --port 8080
```

Use `/api/jobs` for job history, `/api/jobs/{job_id}` for progress, `/api/jobs/{job_id}/qa` for QA, `/api/jobs/{job_id}/output` for output video, and `/ws/jobs/{job_id}` for progress websocket snapshots.
