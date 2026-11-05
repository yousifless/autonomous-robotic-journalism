"""Mission brief (background context) loading with hot reload.

The mission brief is a plain-text file describing the subjects of interest,
the newsworthiness rubric, editorial priorities and safety constraints. It is
injected verbatim into the controller prompt. Editing the file while the
system runs takes effect on the next decision.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional


class MissionBrief:
    def __init__(self, path: str = "", inline_text: str = "") -> None:
        self.path = os.path.expanduser(path) if path else ""
        self._inline = inline_text.strip()
        self._cached = self._inline
        self._mtime: Optional[float] = None

    def text(self) -> str:
        """Current brief; re-read from disk if the file changed."""
        if not self.path:
            return self._cached
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return self._cached
        if self._mtime is None or mtime != self._mtime:
            with open(self.path, "r", encoding="utf-8") as f:
                self._cached = f.read().strip()
            self._mtime = mtime
        return self._cached

    def changed_since(self, digest: str) -> bool:
        return self.digest() != digest

    def digest(self) -> str:
        return hashlib.sha1(self.text().encode("utf-8")).hexdigest()[:12]

    def summary(self, max_chars: int = 300) -> str:
        """First substantive paragraph of the brief, used as the CLIP text query for S_brief.

        CLIP's text encoder accepts at most 77 tokens, so the brief's opening
        sentence(s), which state the mission, are used rather than the whole
        rubric.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", self.text()) if p.strip()]
        # Skip heading lines such as "OFFICE FURNITURE MISSION" (no sentence punctuation).
        for paragraph in paragraphs:
            if re.search(r"[.!?]", paragraph):
                return paragraph[:max_chars]
        return (paragraphs[0] if paragraphs else "")[:max_chars]
