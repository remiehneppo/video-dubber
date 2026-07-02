from __future__ import annotations

import unicodedata
import re
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SemanticMetrics:
    expected_text: str
    transcript: str
    cer: float
    token_recall: float

    def ok(self, *, max_cer: float, min_token_recall: float) -> bool:
        return self.cer <= max_cer and self.token_recall >= min_token_recall

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_DIGIT_WORDS = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

_TOKEN_ALIASES = {
    "doctor": ["dr"],
    "ích": ["x"],
    "dx": ["d", "x"],
    "dt": ["d", "t"],
    "dm": ["d", "m"],
    "dy": ["d", "y"],
    "di": ["d", "y"],
    "pt": ["d", "t"],
    "bx": ["d", "x"],
    "decimet": ["d", "m"],
    "mét": ["m"],
    "met": ["m"],
    "dây": ["giây"],
    "triều": ["chiều"],
    "đảo": ["đạo"],
}


def normalize_semantic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = _expand_math_notation(normalized)
    normalized = "".join(
        character if not unicodedata.category(character).startswith(("P", "S")) else " "
        for character in normalized
    )
    return " ".join(_normalize_math_tokens(normalized.split()))


def _expand_math_notation(text: str) -> str:
    text = text.replace("½", " một phần hai ")
    text = text.replace("×", " nhân ")
    text = text.replace("/", " trên ")
    text = re.sub(r"([a-zà-ỹ])²", r"\1 bình phương", text)
    text = re.sub(r"([a-zà-ỹ])³", r"\1 lập phương", text)
    text = re.sub(r"(\d+)²", r"\1 bình phương", text)
    text = re.sub(r"(\d+)³", r"\1 lập phương", text)
    text = re.sub(r"(\d+)m\b", r"\1 m", text)
    return text


def _normalize_math_tokens(tokens: list[str]) -> list[str]:
    semantic_tokens: list[str] = []
    for index, token in enumerate(tokens):
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        previous_token = tokens[index - 1] if index > 0 else ""
        if token == "sắp" and next_token == "xỉ":
            semantic_tokens.append("xấp")
            continue
        digit_pr = re.fullmatch(r"(\d+)pr", token)
        if digit_pr is not None:
            semantic_tokens.extend([_normalize_digit(digit_pr.group(1)), "pi", "r"])
            continue
        digit_p = re.fullmatch(r"(\d+)p", token)
        if digit_p is not None:
            semantic_tokens.extend([_normalize_digit(digit_p.group(1)), "pi"])
            continue
        if token == "pr":
            semantic_tokens.extend(["pi", "r"])
            continue
        if token in {"drx", "trx"}:
            semantic_tokens.extend(["d", "r", "ít"])
            continue
        if token == "x" and _is_math_factor_token(previous_token) and _is_math_factor_token(next_token):
            semantic_tokens.append("nhân")
            continue
        if re.fullmatch(r"\d+", token):
            semantic_tokens.append(_normalize_digit(token))
            continue
        alias = _TOKEN_ALIASES.get(token)
        if alias is not None:
            semantic_tokens.extend(alias)
            continue
        semantic_tokens.append(token)
    return semantic_tokens


def _is_math_factor_token(token: str) -> bool:
    return bool(
        token in {"pi", "r", "dr", "dx", "dy", "dt", "dm", "pr", "drx", "trx"}
        or re.fullmatch(r"\d+", token)
        or re.fullmatch(r"\d+p", token)
        or re.fullmatch(r"\d+pr", token)
    )


def _normalize_digit(token: str) -> str:
    return _DIGIT_WORDS.get(token, token)


def compare_tts_transcript(expected_text: str, transcript: str) -> SemanticMetrics:
    expected = normalize_semantic_text(expected_text)
    actual = normalize_semantic_text(transcript)
    expected_chars = expected.replace(" ", "")
    actual_chars = actual.replace(" ", "")
    cer = _edit_distance(expected_chars, actual_chars) / max(1, len(expected_chars))
    expected_tokens = Counter(expected.split())
    actual_tokens = Counter(actual.split())
    recalled = sum((expected_tokens & actual_tokens).values())
    expected_token_count = sum(expected_tokens.values())
    token_recall = (
        recalled / expected_token_count
        if expected_token_count
        else float(not actual_tokens)
    )
    return SemanticMetrics(
        expected_text=expected,
        transcript=actual,
        cer=round(cer, 6),
        token_recall=round(token_recall, 6),
    )


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]
