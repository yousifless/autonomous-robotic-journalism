"""Real-time semantic frame selection (Sec. III-B, Table I).

A frame is kept only if its CLIP embedding differs enough from the last kept
frame and a minimum time has passed since then. This removes near-duplicate
frames before the expensive description stage. It reduces processing time;
it does not guarantee that discarded frames carried no relevant evidence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SemanticFrameSelector:
    def __init__(self, similarity_threshold: float = 0.87, min_interval_s: float = 0.4,
                 enabled: bool = True) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_interval_s = min_interval_s
        self.enabled = enabled
        self._last_embedding: Optional[np.ndarray] = None
        self._last_time = -float("inf")
        self.seen = 0
        self.kept = 0

    def reset(self) -> None:
        self._last_embedding = None
        self._last_time = -float("inf")
        self.seen = 0
        self.kept = 0

    def consider(self, embedding: Optional[np.ndarray], timestamp_s: float) -> bool:
        """Return True if the frame should be kept."""
        self.seen += 1
        if not self.enabled or embedding is None:
            self.kept += 1
            return True

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        emb = emb / norm if norm > 0 else emb

        if self._last_embedding is None:
            keep = True
        else:
            similarity = float(np.dot(emb, self._last_embedding))
            keep = (similarity < self.similarity_threshold
                    and (timestamp_s - self._last_time) >= self.min_interval_s)
        if keep:
            self._last_embedding = emb
            self._last_time = timestamp_s
            self.kept += 1
        return keep
