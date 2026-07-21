from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("\n", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned


def sentence_split(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def parse_duration_days(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(day|days|week|weeks|hour|hours)", text.lower())
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)
    if "week" in unit:
        return value * 7
    if "hour" in unit:
        return 1 if value >= 24 else 0
    return value
