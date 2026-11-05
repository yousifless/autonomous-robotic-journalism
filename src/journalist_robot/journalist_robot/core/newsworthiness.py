"""Newsworthiness score N_t (Eq. 1 of the paper).

    N_t = w1 * S_news(I_t) + w2 * S_brief(I_t, T_brief) + w3 * S_novelty(I_t, B_t)

* ``S_news``    - intrinsic journalistic value, rated by the controller VLM.
* ``S_brief``   - cosine similarity between the CLIP embeddings of the current
                  view and of the mission brief.
* ``S_novelty`` - one minus the maximum CLIP similarity between the current
                  view and the views stored in a short-term buffer.

This module only does the arithmetic on embeddings; the CLIP model itself
lives in :mod:`journalist_robot.models.clip_encoder`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class NewsworthinessWeights:
    """Weights of Eq. (1). They must be non-negative and sum to 1."""

    intrinsic: float = 0.4   # w1
    brief: float = 0.4       # w2
    novelty: float = 0.2     # w3

    def __post_init__(self) -> None:
        values = (self.intrinsic, self.brief, self.novelty)
        if any(v < 0.0 for v in values):
            raise ValueError(f"Newsworthiness weights must be non-negative, got {values}")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"Newsworthiness weights must sum to 1, got {sum(values):.4f}")


@dataclass(frozen=True)
class NewsworthinessResult:
    score: float        # N_t
    intrinsic: float    # S_news
    brief: float        # S_brief
    novelty: float      # S_novelty


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0.0 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def rescale(value: float, low: float, high: float) -> float:
    """Linearly map ``[low, high]`` to ``[0, 1]`` and clip.

    Raw CLIP image-text cosine similarities occupy a narrow band (typically
    about 0.15-0.35), so ``S_brief`` is rescaled before it is combined with
    the other [0, 1] signals. The band is a configurable prototype setting.
    """
    if high <= low:
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


class NewsworthinessScorer:
    """Stateful scorer: keeps the short-term view buffer B_t for S_novelty."""

    def __init__(
        self,
        weights: NewsworthinessWeights = NewsworthinessWeights(),
        novelty_buffer_size: int = 8,
        brief_similarity_range: Tuple[float, float] = (0.15, 0.35),
    ) -> None:
        self.weights = weights
        self.brief_similarity_range = brief_similarity_range
        self._buffer: Deque[np.ndarray] = deque(maxlen=max(1, novelty_buffer_size))

    def reset(self) -> None:
        self._buffer.clear()

    def brief_relevance(self, image_embedding: np.ndarray, brief_embedding: Optional[np.ndarray]) -> float:
        if brief_embedding is None:
            return 0.0
        low, high = self.brief_similarity_range
        return rescale(cosine_similarity(image_embedding, brief_embedding), low, high)

    def novelty(self, image_embedding: np.ndarray) -> float:
        """Novelty against the buffer *before* the current view is added."""
        if not self._buffer:
            return 1.0
        max_sim = max(cosine_similarity(image_embedding, past) for past in self._buffer)
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))

    def score(
        self,
        intrinsic: float,
        image_embedding: Optional[np.ndarray],
        brief_embedding: Optional[np.ndarray],
    ) -> NewsworthinessResult:
        """Compute N_t and push the current view into the novelty buffer.

        If no image embedding is available (CLIP disabled or failed), the
        brief and novelty terms are dropped and N_t falls back to S_news.
        """
        s_news = float(np.clip(intrinsic, 0.0, 1.0))
        if image_embedding is None:
            return NewsworthinessResult(score=s_news, intrinsic=s_news, brief=0.0, novelty=0.0)

        s_brief = self.brief_relevance(image_embedding, brief_embedding)
        s_novelty = self.novelty(image_embedding)
        self._buffer.append(_unit(image_embedding))

        w = self.weights
        n_t = w.intrinsic * s_news + w.brief * s_brief + w.novelty * s_novelty
        return NewsworthinessResult(score=float(n_t), intrinsic=s_news, brief=s_brief, novelty=s_novelty)


def combine(weights: NewsworthinessWeights, components: Sequence[float]) -> float:
    """Plain Eq. (1) on already-computed components (S_news, S_brief, S_novelty)."""
    s_news, s_brief, s_novelty = components
    return weights.intrinsic * s_news + weights.brief * s_brief + weights.novelty * s_novelty
