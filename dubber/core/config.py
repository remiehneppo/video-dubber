from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from dubber.core.models import ASRChunkingConfig, ASRServiceConfig, DubberConfig, DubbingCueConfig, InputConfig, LLMServiceConfig, MixingConfig, ProjectConfig, RuntimeConfig, SourceNormalizationConfig, SubtitleConfig, TranslationConfig, TranscriptSegmentationConfig, TTSServiceConfig, VadConfig


def load_config(path: Path | str) -> DubberConfig:
    raw = _load_mapping(Path(path))
    project = raw.get("project", {})
    runtime = raw.get("runtime", {})
    input_config = raw.get("input", {})
    translation = raw.get("translation", {})
    source_normalization = raw.get("source_normalization", {})
    dubbing_cues = raw.get("dubbing_cues", {})
    mixing = raw.get("mixing", {})
    subtitles = raw.get("subtitles", {})
    vad = raw.get("vad", {})
    asr_service = raw.get("asr_service", {})
    asr_chunking = raw.get("asr_chunking", {})
    transcript_segmentation = raw.get("transcript_segmentation", {})
    asr_chunking_config = _load_asr_chunking_config(asr_chunking)
    llm_service = raw.get("llm_service", {})
    tts_service = raw.get("tts_service", {})
    return DubberConfig(
        project=ProjectConfig(
            name=str(project.get("name", "video-dubber")),
            output_format=str(project.get("output_format", "mp4")),
            domain=str(project.get("domain", "mathematics")),
            domain_profile=str(project.get("domain_profile", "")),
            output_dir=Path(str(project.get("output_dir", "output"))),
            workspace_dir=Path(str(project.get("workspace_dir", "workspace"))),
        ),
        runtime=RuntimeConfig(
            max_parallel_jobs=_positive_int(runtime, "max_parallel_jobs", 1),
            asr_concurrency=_positive_int(runtime, "asr_concurrency", 4),
            llm_concurrency=_positive_int(runtime, "llm_concurrency", 2),
            tts_concurrency=_positive_int(runtime, "tts_concurrency", 4),
            retry_max_attempts=int(runtime.get("retry_max_attempts", 3)),
            retry_backoff_sec=int(runtime.get("retry_backoff_sec", 2)),
            request_timeout_sec=int(runtime.get("request_timeout_sec", 120)),
        ),
        input=InputConfig(
            allowed_extensions=list(input_config.get("allowed_extensions", [".mp4", ".mkv", ".mov"])),
            max_file_size_mb=int(input_config.get("max_file_size_mb", 4096)),
            max_duration_minutes=int(input_config.get("max_duration_minutes", 180)),
        ),
        translation=TranslationConfig(
            glossary_review=bool(translation.get("glossary_review", True)),
            generate_spoken_text=bool(translation.get("generate_spoken_text", True)),
            min_context_words=int(translation.get("min_context_words", 120)),
            max_context_words=int(translation.get("max_context_words", 350)),
            context_overlap_words=int(translation.get("context_overlap_words", 40)),
            target_segment_count=int(translation.get("target_segment_count", 6)),
        ),
        source_normalization=SourceNormalizationConfig(
            llm_adjudication=bool(source_normalization.get("llm_adjudication", False)),
        ),
        dubbing_cues=DubbingCueConfig(
            target_duration_ms=int(dubbing_cues.get("target_duration_ms", 4_000)),
            min_duration_ms=int(dubbing_cues.get("min_duration_ms", 1_500)),
            max_duration_ms=int(dubbing_cues.get("max_duration_ms", 6_000)),
        ),
        mixing=MixingConfig(
            original_ducking_db=float(mixing.get("original_ducking_db", -22.0)),
            tts_boost_db=float(mixing.get("tts_boost_db", 8.0)),
            final_loudness_normalization=bool(mixing.get("final_loudness_normalization", True)),
        ),
        subtitles=SubtitleConfig(
            enabled=bool(subtitles.get("enabled", False)),
            mode=str(subtitles.get("mode", "burn_in")),
            output_sidecar=bool(subtitles.get("output_sidecar", True)),
            max_height_ratio=float(subtitles.get("max_height_ratio", 0.10)),
            background_opacity=float(subtitles.get("background_opacity", 0.50)),
            font_family=str(subtitles.get("font_family", "Arial")),
            font_size_ratio=float(subtitles.get("font_size_ratio", 0.026)),
            bottom_margin_ratio=float(subtitles.get("bottom_margin_ratio", 0.035)),
            source_enabled=bool(subtitles.get("source_enabled", True)),
            translation_enabled=bool(subtitles.get("translation_enabled", True)),
            max_cue_duration_ms=int(subtitles.get("max_cue_duration_ms", 4_000)),
            min_cue_duration_ms=int(subtitles.get("min_cue_duration_ms", 700)),
            max_chars_per_line=int(subtitles.get("max_chars_per_line", 48)),
        ),
        vad=VadConfig(
            mode=_vad_mode(vad),
            frame_ms=int(vad.get("frame_ms", 100)),
            threshold_ratio=float(vad.get("threshold_ratio", 0.08)),
            min_speech_duration_ms=int(vad.get("min_speech_duration_ms", 700)),
            target_min_chunk_ms=int(vad.get("target_min_chunk_ms", 20_000)),
            preferred_max_chunk_ms=int(vad.get("preferred_max_chunk_ms", 45_000)),
            hard_max_chunk_ms=int(vad.get("hard_max_chunk_ms", 90_000)),
            silence_merge_threshold_ms=int(vad.get("silence_merge_threshold_ms", 2_500)),
            context_padding_ms=int(vad.get("context_padding_ms", 1_500)),
            soft_split_allowed=bool(vad.get("soft_split_allowed", False)),
            silero_model_path=Path(str(vad.get("silero_model_path", "models/silero_vad.onnx"))),
            silero_model_url=str(vad.get("silero_model_url", "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx")),
            silero_auto_download=bool(vad.get("silero_auto_download", True)),
            silero_threshold=float(vad.get("silero_threshold", 0.5)),
            min_silence_duration_ms=int(vad.get("min_silence_duration_ms", 500)),
            speech_padding_ms=int(vad.get("speech_padding_ms", 250)),
            merge_gap_ms=int(vad.get("merge_gap_ms", 300)),
        ),
        asr_service=ASRServiceConfig(
            provider=str(asr_service.get("provider", "openai_compatible")),
            base_url=str(asr_service.get("base_url", "")),
            api_key=str(asr_service.get("api_key", "")),
            model=str(asr_service.get("model", "whisper-1")),
            language=str(asr_service.get("language", "en")),
            timestamp_mode=str(asr_service.get("timestamp_mode", "prefer_word")),
            require_timestamps=bool(asr_service.get("require_timestamps", True)),
            require_word_timestamps=bool(asr_service.get("require_word_timestamps", True)),
            allow_chunk_text_fallback=bool(asr_service.get("allow_chunk_text_fallback", False)),
            vad_filter=bool(asr_service.get("vad_filter", False)),
        ),
        asr_chunking=asr_chunking_config,
        transcript_segmentation=TranscriptSegmentationConfig(
            target_min_segment_ms=int(transcript_segmentation.get("target_min_segment_ms", 8_000)),
            preferred_max_segment_ms=int(transcript_segmentation.get("preferred_max_segment_ms", 25_000)),
            max_segment_ms=int(transcript_segmentation.get("max_segment_ms", 45_000)),
            min_pause_split_ms=int(transcript_segmentation.get("min_pause_split_ms", 600)),
            prefer_punctuation_split=bool(transcript_segmentation.get("prefer_punctuation_split", True)),
        ),
        llm_service=LLMServiceConfig(
            provider=str(llm_service.get("provider", "openai_compatible")),
            base_url=str(llm_service.get("base_url", "")),
            api_key=str(llm_service.get("api_key", "")),
            model=str(llm_service.get("model", "gpt-4o-mini")),
            temperature=float(llm_service.get("temperature", 0.3)),
        ),
        tts_service=TTSServiceConfig(
            provider=str(tts_service.get("provider", "openai_compatible")),
            base_url=str(tts_service.get("base_url", "")),
            api_key=str(tts_service.get("api_key", "")),
            model=str(tts_service.get("model", "tts-1")),
            voice=str(tts_service.get("voice", "nova")),
            quality_retry_attempts=int(tts_service.get("quality_retry_attempts", 3)),
            rephrase_attempts=int(tts_service.get("rephrase_attempts", 2)),
            max_speedup_ratio=float(tts_service.get("max_speedup_ratio", 1.3)),
            min_rms=float(tts_service.get("min_rms", 500)),
            silence_rms_threshold=int(tts_service.get("silence_rms_threshold", 120)),
            max_edge_silence_ms=int(tts_service.get("max_edge_silence_ms", 1200)),
            max_internal_silence_ms=int(tts_service.get("max_internal_silence_ms", 2500)),
            clipping_peak_threshold=int(tts_service.get("clipping_peak_threshold", 32760)),
            max_clipped_sample_ratio=float(tts_service.get("max_clipped_sample_ratio", 0.001)),
            clause_pause_threshold_ms=int(tts_service.get("clause_pause_threshold_ms", 700)),
            max_overflow_ms=int(tts_service.get("max_overflow_ms", 6_000)),
            overflow_reserve_ms=int(tts_service.get("overflow_reserve_ms", 120)),
            start_delay_ms=int(tts_service.get("start_delay_ms", 0)),
            retained_edge_silence_ms=int(tts_service.get("retained_edge_silence_ms", 100)),
            semantic_max_cer=float(tts_service.get("semantic_max_cer", 0.25)),
            semantic_min_token_recall=float(tts_service.get("semantic_min_token_recall", 0.85)),
            semantic_retry_attempts=int(tts_service.get("semantic_retry_attempts", 3)),
        ),
    )


def _vad_mode(values: dict[str, Any]) -> str:
    mode = str(values.get("mode", "asr_context_chunks"))
    allowed = {"asr_context_chunks", "silero_vad"}
    if mode not in allowed:
        raise ValueError("vad.mode must be one of: asr_context_chunks, silero_vad")
    return mode


def _load_asr_chunking_config(values: dict[str, Any]) -> ASRChunkingConfig:
    config = ASRChunkingConfig(
        enabled=bool(values.get("enabled", True)),
        max_chunk_duration_ms=_positive_int_field(values, "max_chunk_duration_ms", 60_000, "asr_chunking"),
        initial_silence_ms=_positive_int_field(values, "initial_silence_ms", 5_000, "asr_chunking"),
        min_silence_ms=_positive_int_field(values, "min_silence_ms", 500, "asr_chunking"),
        silence_step_ms=_positive_int_field(values, "silence_step_ms", 500, "asr_chunking"),
        trailing_silence_cap_ms=_positive_int_field(values, "trailing_silence_cap_ms", 5_000, "asr_chunking"),
    )
    if config.initial_silence_ms < config.min_silence_ms:
        raise ValueError("asr_chunking.initial_silence_ms must be >= asr_chunking.min_silence_ms")
    return config


def _positive_int(values: dict[str, Any], name: str, default: int) -> int:
    return _positive_int_field(values, name, default, "runtime")


def _positive_int_field(values: dict[str, Any], name: str, default: int, section: str) -> int:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{section}.{name} must be an integer >= 1, got {value!r}")
    return value


def _load_mapping(path: Path) -> dict[str, Any]:
    text = os.path.expandvars(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")
    return loaded


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and raw_line.rstrip().endswith(":"):
            section_name = raw_line.rstrip()[:-1]
            current_section = {}
            data[section_name] = current_section
            continue
        if current_section is None or ":" not in raw_line:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = raw_line.strip().split(":", 1)
        current_section[key] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
