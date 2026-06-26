# Video Dubber Domain Context

## Core Terms

- **Job workspace**: File-based directory for one video dubbing job. Contains state, manifest, copied input, audio, raw provider responses, artifacts, TTS audio, output, and logs.
- **Artifact manifest**: Source of truth for produced files, checksums, stage ownership, and schema versions.
- **Checkpoint**: Persisted job or segment progress used for resume and rerun.
- **Stage**: Ordered job step that reads prior artifacts and publishes one or more new artifacts.
- **Commentary Mode**: Output mode where Vietnamese speech overlays ducked original speech while background audio remains.
- **Glossary review**: Human pause where extracted terms are edited and locked before translation.
- **Provider mode**: Runtime selection between mock local behavior and OpenAI-compatible ASR, LLM, and TTS adapters.
