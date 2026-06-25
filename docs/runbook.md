# Video Dubber Runbook

## Glossary review

Symptom: job status is `waiting_review` and current stage is `glossary`.

Action:

1. Open `workspace/{job_id}/artifacts/glossary.draft.json`.
2. Review terms and translations.
3. Save the reviewed file as `workspace/{job_id}/artifacts/glossary.locked.json` with `status: locked` and locked terms.
4. Run `python3 cli.py resume --job {job_id}`.

## Crash during TTS

Symptom: job status is `failed`, current stage is `tts`, and `last_error` contains a provider or simulated crash message.

Action:

1. Inspect `workspace/{job_id}/artifacts/tts_segments.v1.json`.
2. Confirm completed segments remain marked `completed`.
3. Run `python3 cli.py resume --job {job_id}` to recover the TTS stage and render output.

## Corrupt artifact

Symptom: `python3 cli.py validate --job {job_id}` reports an invalid artifact such as `translated.v1`.

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
