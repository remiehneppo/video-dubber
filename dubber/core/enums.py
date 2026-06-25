from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_FAILED = "partial_failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_REVIEW = "waiting_review"


class StageName(str, Enum):
    JOB_INIT = "job_init"
    AUDIO_EXTRACT = "audio_extract"
    VAD = "vad"
    ASR = "asr"
    GLOSSARY = "glossary"
    TRANSLATION = "translation"
    TTS = "tts"
    MIXING = "mixing"

