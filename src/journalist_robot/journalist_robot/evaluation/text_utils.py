"""Sentence splitting shared by the evaluation signals."""

from __future__ import annotations

import re
from typing import List

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str, min_chars: int = 15) -> List[str]:
    sentences = []
    for paragraph in text.split("\n"):
        for s in _SPLIT.split(paragraph.strip()):
            s = s.strip()
            if len(s) >= min_chars:
                sentences.append(s)
    return sentences


def word_count(text: str) -> int:
    return len(text.split())
