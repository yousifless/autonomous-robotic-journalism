import numpy as np
import pytest

from journalist_robot.core.newsworthiness import (NewsworthinessScorer, NewsworthinessWeights, combine,
                                                  cosine_similarity, rescale)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        NewsworthinessWeights(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        NewsworthinessWeights(1.2, -0.2, 0.0)


def test_equation_1_with_paper_weights():
    w = NewsworthinessWeights()          # 0.4 / 0.4 / 0.2
    assert combine(w, (1.0, 0.5, 0.0)) == pytest.approx(0.6)
    assert combine(w, (0.5, 0.5, 1.0)) == pytest.approx(0.6)


def test_rescale():
    assert rescale(0.25, 0.15, 0.35) == pytest.approx(0.5)
    assert rescale(0.0, 0.15, 0.35) == 0.0
    assert rescale(1.0, 0.15, 0.35) == 1.0


def test_novelty_drops_for_repeated_view():
    scorer = NewsworthinessScorer(brief_similarity_range=(0.0, 1.0))
    view = np.array([1.0, 0.0, 0.0])
    brief = np.array([1.0, 0.0, 0.0])
    first = scorer.score(0.5, view, brief)
    second = scorer.score(0.5, view, brief)
    assert first.novelty == pytest.approx(1.0)
    assert second.novelty == pytest.approx(0.0, abs=1e-6)
    assert first.brief == pytest.approx(1.0)
    assert first.score == pytest.approx(0.4 * 0.5 + 0.4 * 1.0 + 0.2 * 1.0)
    assert second.score < first.score


def test_falls_back_to_intrinsic_without_embeddings():
    res = NewsworthinessScorer().score(0.8, None, None)
    assert res.score == pytest.approx(0.8)
    assert (res.brief, res.novelty) == (0.0, 0.0)


def test_cosine_similarity():
    assert cosine_similarity(np.array([1, 0]), np.array([0, 2])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1, 1]), np.array([2, 2])) == pytest.approx(1.0)
