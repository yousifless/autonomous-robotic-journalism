import numpy as np
import pytest

from journalist_robot.analysis.decision_metrics import alignment, discretise, divergence
from journalist_robot.evaluation.acceptance import AcceptanceWeights, EvaluationReport, SignalResult
from journalist_robot.journalism.frame_selector import SemanticFrameSelector
from journalist_robot.journalism.view_check import parse_view_check
from journalist_robot.models.article_writer import parse_article, strip_reasoning


# ------------------------------------------------------------ frame selection
def test_selector_drops_near_duplicates_and_respects_interval():
    sel = SemanticFrameSelector(similarity_threshold=0.87, min_interval_s=0.4)
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert sel.consider(a, 0.0) is True       # first frame always kept
    assert sel.consider(a, 1.0) is False      # identical view
    assert sel.consider(b, 0.1) is False      # different view but too soon
    assert sel.consider(b, 1.0) is True
    assert (sel.seen, sel.kept) == (4, 2)


def test_selector_disabled_keeps_everything():
    sel = SemanticFrameSelector(enabled=False)
    assert all(sel.consider(np.array([1.0, 0.0]), t) for t in (0.0, 0.1, 0.2))


# ----------------------------------------------------------------- view check
def test_view_check_parsing():
    r = parse_view_check('{"view_usable": false, "on_topic": true, '
                         '"suggested_action": "rotate_left", "guidance": "Turn toward the presenter."}')
    assert r.needs_feedback
    assert r.suggested_action == "ROTATE_LEFT"
    assert "presenter" in r.prompt_text()


def test_unparseable_view_check_does_not_steer():
    r = parse_view_check("no json here")
    assert not r.needs_feedback
    assert r.suggested_action == "NONE"


# ------------------------------------------------------------ article parsing
def test_parse_article_format_and_reasoning_trace():
    raw = "<think>plan the story</think>\nHEADLINE: Robot Built in Lab\nLEAD: Researchers assemble a robot.\nBODY:\nPara 1.\n\nPara 2."
    art = parse_article(raw)
    assert art.headline == "Robot Built in Lab"
    assert art.lead == "Researchers assemble a robot."
    assert art.body.startswith("Para 1.")
    assert "think" not in art.raw


def test_parse_article_fallback():
    art = parse_article("A Headline\nSome body text.")
    assert art.headline == "A Headline"
    assert art.body == "Some body text."


def test_strip_reasoning():
    assert strip_reasoning("<think>x</think> answer") == "answer"


# ------------------------------------------------------------ acceptance (Eq. 4)
def test_acceptance_score_uses_paper_weights():
    report = EvaluationReport(SignalResult(1.0), SignalResult(0.8), SignalResult(0.5), SignalResult(0.5),
                              AcceptanceWeights(), threshold=0.85)
    assert report.score == pytest.approx(0.35 * 1.0 + 0.25 * 0.8 + 0.20 * 0.5 + 0.20 * 0.5)
    assert report.accepted is False
    assert report.as_dict()["accepted"] is False


def test_acceptance_weights_validation():
    with pytest.raises(ValueError):
        AcceptanceWeights(0.5, 0.5, 0.5, 0.5)


# ------------------------------------------------------------ navigation metrics
def test_discretisation_thresholds():
    assert discretise(0.0, 0.3) == "Left"
    assert discretise(0.0, -0.3) == "Right"
    assert discretise(0.2, 0.1) == "Forward"
    assert discretise(0.0, 0.1) == "Stop"
    assert discretise(0.2, 0.3, precedence="rotation") == "Left"
    assert discretise(0.2, 0.3, precedence="translation") == "Forward"


def test_alignment_and_divergence():
    ref = ["Left", "Forward", "Stop", "Right"]
    cand = ["Left", "Forward", "Right", "Right"]
    assert alignment(ref, cand, budget=4) == pytest.approx(0.75)
    assert divergence(ref, cand, budget=4) == pytest.approx(0.25)
    assert alignment(ref, cand, budget=2) == pytest.approx(1.0)
