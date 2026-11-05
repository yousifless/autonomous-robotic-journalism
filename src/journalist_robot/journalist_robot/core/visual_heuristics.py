"""Lightweight image heuristics used next to the VLM.

* ``reflex_metrics``  - edge density / entropy in the centre and sides of the
                        view, used to suppress forward motion into a view that
                        looks like a blank wall and to choose a turn direction.
* ``motion_metric``   - frame-difference measure of scene dynamics, used to
                        adapt the sampling temperature and as a prompt hint.

These are heuristics, not obstacle detection.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np


def _roi(image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(round(width * x))
    y0 = int(round(height * y))
    x1 = min(width, x0 + max(1, int(round(width * w))))
    y1 = min(height, y0 + max(1, int(round(height * h))))
    return image[y0:y1, x0:x1]


def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _entropy(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(1.0, float(hist.sum()))
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / 8.0)  # normalised to [0, 1]


def reflex_metrics(frames: List[np.ndarray],
                   blocked_edge_density: float = 0.02,
                   blocked_entropy: float = 0.35) -> Dict[str, object]:
    """Visual cues for the reflex gate, averaged over the last (up to) 4 frames."""
    if not frames:
        return {
            "center_edge_density": 0.0, "center_entropy": 0.0,
            "left_edge_density": 0.0, "right_edge_density": 0.0,
            "center_blocked": False, "preferred_turn": None,
            "visual_clarity": 0.0,
        }

    center_e, center_h, left_e, right_e = [], [], [], []
    for frame in frames[-4:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center = _roi(gray, 0.35, 0.35, 0.30, 0.30)
        center_e.append(_edge_density(center))
        center_h.append(_entropy(center))
        left_e.append(_edge_density(_roi(gray, 0.00, 0.0, 0.33, 1.0)))
        right_e.append(_edge_density(_roi(gray, 0.67, 0.0, 0.33, 1.0)))

    ce, ch = float(np.mean(center_e)), float(np.mean(center_h))
    le, re_ = float(np.mean(left_e)), float(np.mean(right_e))

    preferred_turn: Optional[str] = None
    if abs(le - re_) > 0.01:
        # Turn towards the side with more structure (the emptier side is more
        # likely a wall or a featureless obstacle).
        preferred_turn = "LEFT" if le > re_ else "RIGHT"

    return {
        "center_edge_density": ce,
        "center_entropy": ch,
        "left_edge_density": le,
        "right_edge_density": re_,
        "center_blocked": bool(ce < blocked_edge_density and ch < blocked_entropy),
        "preferred_turn": preferred_turn,
        "visual_clarity": float(min(1.0, (ce * 10.0 + ch) / 2.0)),
    }


def motion_metric(frames: List[np.ndarray]) -> float:
    """0.7 * mean absolute pixel difference + 0.3 * edge-change ratio."""
    if len(frames) < 2:
        return 0.0
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    pixel, edge = [], []
    for prev, cur in zip(gray[:-1], gray[1:]):
        pixel.append(float(np.mean(cv2.absdiff(cur, prev))))
        diff = cv2.absdiff(cv2.Canny(prev, 50, 150), cv2.Canny(cur, 50, 150))
        edge.append(float(np.count_nonzero(diff)) / float(diff.size))
    return 0.7 * float(np.mean(pixel)) + 0.3 * float(np.mean(edge))


class ForwardConsensus:
    """Forward motion requires the centre to look clear in 2 of the last 3 cycles."""

    def __init__(self, window: int = 3, required: int = 2) -> None:
        self._history: Deque[bool] = deque(maxlen=window)
        self._required = required

    def update(self, center_clear: bool) -> bool:
        self._history.append(center_clear)
        return len(self._history) >= self._required and sum(self._history) >= self._required


def sampling_temperature(motion: float, reflex: Dict[str, object]) -> float:
    """Lower temperature when the view is blocked or clear; higher when ambiguous."""
    if reflex.get("center_blocked"):
        return 0.4
    base = 0.5 if motion < 4.0 else 0.8
    clarity = float(reflex.get("visual_clarity", 0.5))
    if clarity > 0.7:
        return max(0.1, base - 0.2)
    if clarity < 0.3:
        return min(0.9, base + 0.2)
    return base
