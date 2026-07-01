from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dubber.core.enums import JobStatus, StageName, StageStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "video-dubber"
    output_format: str = "mp4"
    domain: str = "mathematics"
    domain_profile: str = ""
    output_dir: Path = Path("output")
    workspace_dir: Path = Path("workspace")


@dataclass(frozen=True)
class RuntimeConfig:
    max_parallel_jobs: int = 1
    asr_concurrency: int = 4
    llm_concurrency: int = 2
    tts_concurrency: int = 4
    retry_max_attempts: int = 3
    retry_backoff_sec: int = 2
    request_timeout_sec: int = 120


@dataclass(frozen=True)
class InputConfig:
    allowed_extensions: list[str] = field(default_factory=lambda: [".mp4", ".mkv", ".mov"])
    max_file_size_mb: int = 4096
    max_duration_minutes: int = 180


@dataclass(frozen=True)
class VadConfig:
    mode: str = "asr_context_chunks"
    frame_ms: int = 100
    threshold_ratio: float = 0.08
    min_duration_ms: int = 900
    max_duration_ms: int = 60_000
    min_speech_duration_ms: int = 700
    target_min_chunk_ms: int = 20_000
    preferred_max_chunk_ms: int = 45_000
    hard_max_chunk_ms: int = 90_000
    silence_merge_threshold_ms: int = 2_500
    context_padding_ms: int = 1_500
    soft_split_allowed: bool = False
    silero_model_path: Path = Path("models/silero_vad.onnx")
    silero_model_url: str = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
    silero_auto_download: bool = True
    silero_threshold: float = 0.5
    min_silence_duration_ms: int = 500
    speech_padding_ms: int = 250
    max_vad_chunk_ms: int = 30_000
    merge_gap_ms: int = 300


@dataclass(frozen=True)
class MixingConfig:
    original_ducking_db: float = -22.0
    tts_boost_db: float = 8.0
    final_loudness_normalization: bool = True


@dataclass(frozen=True)
class SubtitleConfig:
    enabled: bool = False
    mode: str = "burn_in"
    output_sidecar: bool = True
    max_height_ratio: float = 0.10
    background_opacity: float = 0.50
    font_family: str = "Arial"
    font_size_ratio: float = 0.026
    bottom_margin_ratio: float = 0.035
    source_enabled: bool = True
    translation_enabled: bool = True
    max_cue_duration_ms: int = 4_000
    min_cue_duration_ms: int = 700
    max_chars_per_line: int = 48


@dataclass(frozen=True)
class TranslationConfig:
    glossary_review: bool = True
    min_context_words: int = 120
    max_context_words: int = 350
    context_overlap_words: int = 40
    target_segment_count: int = 6


@dataclass(frozen=True)
class SourceNormalizationConfig:
    llm_adjudication: bool = False


@dataclass(frozen=True)
class DubbingCueConfig:
    target_duration_ms: int = 4_000
    min_duration_ms: int = 1_500
    max_duration_ms: int = 6_000


@dataclass(frozen=True)
class ASRServiceConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = "whisper-1"
    language: str = "en"
    timestamp_mode: str = "prefer_word"
    require_timestamps: bool = True
    require_word_timestamps: bool = True
    allow_chunk_text_fallback: bool = False
    vad_filter: bool = False


@dataclass(frozen=True)
class TranscriptSegmentationConfig:
    target_min_segment_ms: int = 8_000
    preferred_max_segment_ms: int = 25_000
    max_segment_ms: int = 45_000
    min_pause_split_ms: int = 600
    prefer_punctuation_split: bool = True


@dataclass(frozen=True)
class LLMServiceConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.3


@dataclass(frozen=True)
class TTSServiceConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = "tts-1"
    voice: str = "nova"
    quality_retry_attempts: int = 3
    rephrase_attempts: int = 2
    max_speedup_ratio: float = 1.3
    min_rms: float = 500
    silence_rms_threshold: int = 120
    max_edge_silence_ms: int = 1200
    max_internal_silence_ms: int = 2500
    clipping_peak_threshold: int = 32760
    max_clipped_sample_ratio: float = 0.001
    clause_pause_threshold_ms: int = 700
    max_overflow_ms: int = 6_000
    overflow_reserve_ms: int = 120
    start_delay_ms: int = 0
    retained_edge_silence_ms: int = 100
    semantic_max_cer: float = 0.25
    semantic_min_token_recall: float = 0.85
    semantic_retry_attempts: int = 3


@dataclass(frozen=True)
class DubberConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    input: InputConfig = field(default_factory=InputConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    source_normalization: SourceNormalizationConfig = field(default_factory=SourceNormalizationConfig)
    dubbing_cues: DubbingCueConfig = field(default_factory=DubbingCueConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr_service: ASRServiceConfig = field(default_factory=ASRServiceConfig)
    transcript_segmentation: TranscriptSegmentationConfig = field(default_factory=TranscriptSegmentationConfig)
    llm_service: LLMServiceConfig = field(default_factory=LLMServiceConfig)
    tts_service: TTSServiceConfig = field(default_factory=TTSServiceConfig)


@dataclass(frozen=True)
class StageProgress:
    status: StageStatus = StageStatus.PENDING
    artifact: str | None = None
    done: int | None = None
    total: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"status": self.status.value}
        if self.artifact is not None:
            data["artifact"] = self.artifact
        if self.done is not None:
            data["done"] = self.done
        if self.total is not None:
            data["total"] = self.total
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StageProgress:
        return cls(
            status=StageStatus(str(data["status"])),
            artifact=str(data["artifact"]) if data.get("artifact") is not None else None,
            done=int(data["done"]) if data.get("done") is not None else None,
            total=int(data["total"]) if data.get("total") is not None else None,
            error=str(data["error"]) if data.get("error") is not None else None,
        )


@dataclass(frozen=True)
class JobState:
    schema_version: str
    job_id: str
    status: JobStatus
    current_stage: StageName
    input_file: Path
    stages: dict[StageName, StageProgress]
    last_error: str | None
    created_at: str
    updated_at: str

    @classmethod
    def create(cls, job_id: str, input_file: Path) -> JobState:
        now = utc_now_iso()
        return cls(
            schema_version="1.0",
            job_id=job_id,
            status=JobStatus.RUNNING,
            current_stage=StageName.JOB_INIT,
            input_file=input_file,
            stages={stage: StageProgress() for stage in StageName},
            last_error=None,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "input_file": str(self.input_file),
            "stages": {stage.value: progress.to_dict() for stage, progress in self.stages.items()},
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> JobState:
        raw_stages = data.get("stages", {})
        if not isinstance(raw_stages, dict):
            raise ValueError("job_state stages must be an object")
        stages = {stage: StageProgress() for stage in StageName}
        for name, progress in raw_stages.items():
            stages[StageName(str(name))] = StageProgress.from_dict(progress)  # type: ignore[arg-type]
        return cls(
            schema_version=str(data["schema_version"]),
            job_id=str(data["job_id"]),
            status=JobStatus(str(data["status"])),
            current_stage=StageName(str(data["current_stage"])),
            input_file=Path(str(data["input_file"])),
            stages=stages,
            last_error=str(data["last_error"]) if data.get("last_error") is not None else None,
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )
