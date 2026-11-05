"""Evidence accumulated for one segment: descriptions, transcript and key frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from journalist_robot.models.transcriber import TranscriptChunk


@dataclass
class DescriptionBatch:
    batch_index: int
    frame_indices: List[int]
    timestamps_s: List[float]
    text: str
    speech: str = ""

    def formatted(self) -> str:
        span = (f"{self.timestamps_s[0]:.1f}s-{self.timestamps_s[-1]:.1f}s"
                if self.timestamps_s else "unknown time")
        return f"--- Frames {self.frame_indices[0]}-{self.frame_indices[-1]} ({span}) ---\n{self.text}"


@dataclass
class SegmentEvidence:
    segment_id: int
    brief: str
    start_time: float
    descriptions: List[DescriptionBatch] = field(default_factory=list)
    transcript: List[TranscriptChunk] = field(default_factory=list)
    key_frames: List[np.ndarray] = field(default_factory=list)      # for evaluation / UI
    key_frame_times: List[float] = field(default_factory=list)
    collection_end_time: Optional[float] = None

    def descriptions_text(self) -> str:
        return "\n\n".join(d.formatted() for d in self.descriptions)

    def transcript_text(self) -> str:
        return "\n".join(c.line() for c in self.transcript)

    def speech_between(self, t0: float, t1: float) -> str:
        lines = [c.line() for c in self.transcript if c.end_s >= t0 and c.start_s <= t1]
        return "\n".join(lines)

    def timeline(self) -> str:
        """Compact list of batch time spans with the first sentence of each description."""
        lines = []
        for d in self.descriptions:
            first = d.text.strip().split(". ")[0][:200]
            t = f"{d.timestamps_s[0]:.1f}s" if d.timestamps_s else "?"
            lines.append(f"- {t}: {first}")
        return "\n".join(lines)

    def description_texts(self) -> List[str]:
        return [d.text for d in self.descriptions]
