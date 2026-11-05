"""Acceptance score S (Eq. 4) and the accept / rewrite decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class AcceptanceWeights:
    factuality: float = 0.35
    relevance: float = 0.25
    consistency: float = 0.20
    language: float = 0.20

    def __post_init__(self) -> None:
        total = self.factuality + self.relevance + self.consistency + self.language
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Acceptance weights must sum to 1, got {total:.4f}")


@dataclass
class SignalResult:
    score: float
    details: Dict[str, float] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    factuality: SignalResult
    relevance: SignalResult
    consistency: SignalResult
    language: SignalResult
    weights: AcceptanceWeights
    threshold: float

    @property
    def score(self) -> float:
        w = self.weights
        return (w.factuality * self.factuality.score + w.relevance * self.relevance.score
                + w.consistency * self.consistency.score + w.language * self.language.score)

    @property
    def accepted(self) -> bool:
        return self.score >= self.threshold

    def problems_text(self, max_items: int = 12) -> str:
        items: List[str] = []
        for name, sig in (("factuality", self.factuality), ("relevance", self.relevance),
                          ("consistency", self.consistency), ("language", self.language)):
            items.extend(f"- [{name}] {p}" for p in sig.problems)
        return "\n".join(items[:max_items])

    def as_dict(self) -> Dict[str, object]:
        def sig(s: SignalResult) -> Dict[str, object]:
            return {"score": s.score, "details": s.details, "problems": s.problems}

        return {
            "acceptance_score": self.score,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "weights": vars(self.weights),
            "factuality": sig(self.factuality),
            "relevance": sig(self.relevance),
            "consistency": sig(self.consistency),
            "language": sig(self.language),
            "note": "Internal model-based heuristic; not an independent quality assessment.",
        }
