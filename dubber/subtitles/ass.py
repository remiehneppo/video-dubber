from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Any

from dubber.core.models import SubtitleConfig


@dataclass(frozen=True)
class SubtitleCue:
    start_ms: int
    end_ms: int
    source_text: str
    translation_text: str


def build_spoken_subtitle_cues(spoken: dict[str, Any]) -> list[SubtitleCue]:
    """Build subtitle cues from the exact text and timing accepted by TTS QA."""
    cues: list[SubtitleCue] = []
    for item in spoken.get("cues", []):
        if not isinstance(item, dict):
            continue
        start_ms = int(item.get("original_start_ms", item.get("target_start_ms", 0)))
        end_ms = int(item.get("original_end_ms", item.get("target_end_ms", start_ms)))
        source_text = str(item.get("source_text", "")).strip()
        display_text = str(item.get("display_text") or item.get("final_text") or "").strip()
        if source_text or display_text:
            cues.append(SubtitleCue(start_ms, max(start_ms + 1, end_ms), source_text, display_text))
    return cues


def build_subtitle_cues(
    transcript: dict[str, Any],
    translated: dict[str, Any],
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    translations_by_id = {
        str(segment.get("segment_id")): str(segment.get("display_text") or segment.get("vi_text") or "").strip()
        for segment in translated.get("segments", [])
        if isinstance(segment, dict)
    }
    cues: list[SubtitleCue] = []
    for segment in transcript.get("segments", []):
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id", ""))
        source_text = str(segment.get("source_text", "")).strip()
        translation_text = translations_by_id.get(segment_id, "")
        if not source_text and not translation_text:
            continue
        segment_cues = _split_segment_cues(
            segment,
            source_text=source_text,
            translation_text=translation_text,
            config=config,
        )
        cues.extend(segment_cues)
    return cues


def render_ass(
    cues: list[SubtitleCue],
    config: SubtitleConfig,
    *,
    video_width: int,
    video_height: int,
) -> str:
    play_res_x = max(1, int(video_width))
    play_res_y = max(1, int(video_height))
    font_size = max(10, int(play_res_y * config.font_size_ratio))
    margin_v = max(4, int(play_res_y * config.bottom_margin_ratio))
    max_subtitle_height = max(1, int(play_res_y * config.max_height_ratio))
    outline = max(1, int(font_size * 0.10))
    shadow = max(0, int(font_size * 0.08))
    back_colour = _ass_alpha_black(config.background_opacity)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Bilingual,{config.font_family},{font_size},&H00FFFFFF,&H00FFFFFF,&H80000000,{back_colour},0,0,0,0,100,100,0,0,3,{outline},{shadow},2,24,24,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        text_lines: list[str] = []
        if config.source_enabled and cue.source_text.strip():
            text_lines.extend(_wrap_text(cue.source_text, config.max_chars_per_line))
        if config.translation_enabled and cue.translation_text.strip():
            text_lines.extend(_wrap_text(cue.translation_text, config.max_chars_per_line))
        if not text_lines:
            continue
        text = r"\N".join(_escape_ass_text(line) for line in text_lines)
        lines.append(
            "Dialogue: 0,"
            f"{_format_ass_time(cue.start_ms)},{_format_ass_time(cue.end_ms)},"
            f"Bilingual,,0,0,0,,{{\\an2\\q2\\pos({play_res_x // 2},{play_res_y - margin_v})\\pbo0}}{text}"
        )
    lines.append("")
    lines.append(f"; MaxSubtitleHeight: {max_subtitle_height}")
    return "\n".join(lines)


def _split_segment_cues(
    segment: dict[str, Any],
    *,
    source_text: str,
    translation_text: str,
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    start_ms = int(segment.get("start_ms", 0))
    end_ms = max(start_ms + config.min_cue_duration_ms, int(segment.get("end_ms", start_ms)))
    words = _valid_words(segment.get("words"))
    if not words or end_ms - start_ms <= config.max_cue_duration_ms:
        return [SubtitleCue(start_ms=start_ms, end_ms=end_ms, source_text=source_text, translation_text=translation_text)]

    word_groups = _split_words_by_duration(words, config.max_cue_duration_ms, config.min_cue_duration_ms)
    translation_parts = _split_translation_for_groups(translation_text, word_groups)
    cues: list[SubtitleCue] = []
    for index, group in enumerate(word_groups):
        cue_start = int(group[0]["start_ms"])
        cue_end = max(cue_start + config.min_cue_duration_ms, int(group[-1]["end_ms"]))
        cue_end = min(cue_end, end_ms)
        cues.append(
            SubtitleCue(
                start_ms=cue_start,
                end_ms=cue_end,
                source_text=" ".join(str(word["text"]).strip() for word in group).strip(),
                translation_text=translation_parts[index] if index < len(translation_parts) else "",
            )
        )
    return cues


def _valid_words(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    words: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        if not text:
            continue
        try:
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_ms <= start_ms:
            continue
        words.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
    return words


def _split_words_by_duration(
    words: list[dict[str, object]],
    max_cue_duration_ms: int,
    min_cue_duration_ms: int,
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for word in words:
        if not current:
            current = [word]
            continue
        candidate_duration = int(word["end_ms"]) - int(current[0]["start_ms"])
        if candidate_duration > max_cue_duration_ms and int(current[-1]["end_ms"]) - int(current[0]["start_ms"]) >= min_cue_duration_ms:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        groups.append(current)
    return groups


def _split_translation_for_groups(text: str, word_groups: list[list[dict[str, object]]]) -> list[str]:
    if not word_groups:
        return []
    text = text.strip()
    if not text:
        return [""] * len(word_groups)
    sentences = _split_sentences(text)
    if len(sentences) == len(word_groups):
        return sentences
    translation_words = text.split()
    if len(word_groups) == 1 or not translation_words:
        return [text]
    total_source_words = sum(len(group) for group in word_groups)
    parts: list[str] = []
    cursor = 0
    for index, group in enumerate(word_groups):
        if index == len(word_groups) - 1:
            part_words = translation_words[cursor:]
        else:
            share = len(group) / max(1, total_source_words)
            take = max(1, round(len(translation_words) * share))
            remaining_groups = len(word_groups) - index - 1
            take = min(take, max(1, len(translation_words) - cursor - remaining_groups))
            part_words = translation_words[cursor:cursor + take]
            cursor += take
        parts.append(" ".join(part_words).strip())
    return parts


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if part.strip()]
    return parts if parts else [text.strip()]


def _wrap_text(text: str, max_chars_per_line: int) -> list[str]:
    max_chars = max(8, int(max_chars_per_line))
    return textwrap.wrap(
        text.strip(),
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [text.strip()]


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .strip()
    )


def _format_ass_time(ms: int) -> str:
    centiseconds = max(0, int(ms)) // 10
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_alpha_black(opacity: float) -> str:
    alpha = int(round((1.0 - min(1.0, max(0.0, opacity))) * 255))
    return f"&H{alpha:02X}000000"
