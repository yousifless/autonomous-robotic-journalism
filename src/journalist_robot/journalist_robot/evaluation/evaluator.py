"""Runs the four signals on a draft and combines them with Eq. (4)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from journalist_robot.evaluation.acceptance import AcceptanceWeights, EvaluationReport
from journalist_robot.evaluation.signals import (ConsistencySignal, FactualitySignal, LanguageSignal,
                                                 RelevanceSignal)


class ArticleEvaluator:
    def __init__(self, clip_encoder, describer, weights: AcceptanceWeights = AcceptanceWeights(),
                 threshold: float = 0.85, device: str = "cuda",
                 language_tool_url: Optional[str] = None, local_files_only: bool = False) -> None:
        self.weights = weights
        self.threshold = threshold
        self.factuality = FactualitySignal(clip_encoder, device=device, local_files_only=local_files_only)
        self.relevance = RelevanceSignal(clip_encoder)
        self.consistency = ConsistencySignal(describer)
        self.language = LanguageSignal(language_tool_url=language_tool_url, device=device)

    def evaluate(self, headline: str, lead: str, body: str,
                 key_frames: Sequence[np.ndarray], descriptions: List[str]) -> EvaluationReport:
        article = f"{headline}\n\n{lead}\n\n{body}".strip()
        return EvaluationReport(
            factuality=self.factuality.evaluate(article, key_frames, descriptions),
            relevance=self.relevance.evaluate(article, key_frames),
            consistency=self.consistency.evaluate(article, key_frames),
            language=self.language.evaluate(headline, lead, body),
            weights=self.weights,
            threshold=self.threshold,
        )
