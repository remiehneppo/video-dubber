# Video Dubber

CLI-first Vietnamese commentary dubbing pipeline for educational videos, lectures, podcasts, and explainer content.

The current implementation is a modular monolith with file-based jobs, manifest-tracked artifacts, checkpoint/resume, a mock provider mode for local end-to-end runs, OpenAI-compatible provider adapters, and a FastAPI monitor.

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

The job pauses with `waiting_review` and writes `artifacts/glossary.draft.json`. Review it, save `artifacts/glossary.locked.json`, then run:

```bash
dubber resume --job <job_id>
# or: python3 cli.py resume --job <job_id>
```

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

## Web Monitor

```bash
dubber web --workspace workspace --host 127.0.0.1 --port 8080
# or: python3 cli.py web --workspace workspace --host 127.0.0.1 --port 8080
```

API endpoints:

- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/qa`
- `GET /api/jobs/{job_id}/output`
- `WS /ws/jobs/{job_id}`

## Test

```bash
pytest -q
```
