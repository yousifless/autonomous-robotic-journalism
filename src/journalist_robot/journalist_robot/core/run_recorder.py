"""On-disk record of a run: decisions, segments, descriptions, articles.

Layout (one folder per launch)::

    <output_root>/<run_id>/
        decisions.jsonl                 one line per control decision
        decision_frames/000123.jpg      input frame of each decision (optional)
        segment_001/
            frames/000000.jpg           selected frames sent to the describer
            descriptions.jsonl          one line per description batch
            transcript.txt              timestamped speech transcript
            feedback.jsonl              captured-view checks
            article.md                  final draft (headline, lead, body)
            article.json                draft + acceptance scores + timings
            drafts/draft_0.md ...       every draft, including rejected ones

Every file is written incrementally so a crash does not lose the run.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RunRecorder:
    def __init__(self, output_root: str, run_id: Optional[str] = None) -> None:
        self.root = os.path.join(os.path.expanduser(output_root), run_id or make_run_id())
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.Lock()

    # -- generic helpers --------------------------------------------------
    def path(self, *parts: str) -> str:
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def append_jsonl(self, relative_path: str, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock, open(self.path(relative_path), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_text(self, relative_path: str, text: str) -> str:
        p = self.path(relative_path)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    def write_json(self, relative_path: str, obj: Dict[str, Any]) -> str:
        p = self.path(relative_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        return p

    def write_image(self, relative_path: str, image_bgr: np.ndarray, quality: int = 90) -> str:
        p = self.path(relative_path)
        cv2.imwrite(p, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return p

    # -- segment helpers ---------------------------------------------------
    @staticmethod
    def segment_dir(segment_id: int) -> str:
        return f"segment_{segment_id:03d}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
