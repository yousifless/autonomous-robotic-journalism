"""Safety-relevant behaviour of the parser, the control policy and the writer output handling."""

import math

import pytest

from journalist_robot.core.control_policy import (ControlPolicy, ExplorationState, PolicyConfig, PolicyInput,
                                                  VelocityLimits, clamp)
from journalist_robot.core.json_utils import parse_controller_output
from journalist_robot.evaluation.signals import FactualitySignal
from journalist_robot.models.article_writer import parse_article


# ------------------------------------------------------------ VLM number handling
@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_velocities_become_zero(raw):
    out = parse_controller_output(f'{{"linear_x": {raw}, "angular_z": {raw}, "confidence": 0.9}}')
    assert out["linear_x"] == 0.0
    assert out["angular_z"] == 0.0


def test_scores_are_clipped_and_percentages_rescaled():
    out = parse_controller_output('{"linear_x": 0.1, "confidence": 85, "newsworthiness_score": NaN}')
    assert out["confidence"] == pytest.approx(0.85)
    assert out["newsworthiness_score"] == 0.0
    out = parse_controller_output('{"confidence": 1e9, "newsworthiness_score": -3}')
    assert out["confidence"] == 0.0
    assert out["newsworthiness_score"] == 0.0


def test_fallback_parser_is_sanitized_too():
    out = parse_controller_output('"linear_x": 0.2, "confidence": 90, truncated')
    assert out["parsed_with_fallback"] is True
    assert out["confidence"] == pytest.approx(0.9)


def test_clamp_maps_nan_to_zero():
    assert clamp(math.nan, -0.4, 0.4) == 0.0
    assert clamp(math.inf, -0.4, 0.4) == 0.0


# ------------------------------------------------------------ no blind reversing
def _apply(policy, linear_x):
    return policy.apply(PolicyInput(linear_x=linear_x, angular_z=0.0, confidence=0.9, newsworthiness=0.9),
                        ExplorationState(), now=0.0)


def test_reverse_is_disabled_by_default():
    policy = ControlPolicy(PolicyConfig(limits=VelocityLimits(0.4, 0.5)))
    assert _apply(policy, -0.3).linear_x == 0.0


def test_reverse_can_be_enabled():
    policy = ControlPolicy(PolicyConfig(limits=VelocityLimits(0.4, 0.5), allow_reverse=True))
    assert _apply(policy, -0.3).linear_x < 0.0


# ------------------------------------------------------------ writer output
def test_unfinished_reasoning_gives_empty_draft():
    draft = parse_article("<think>Let me plan the article. First the headline")
    assert (draft.headline, draft.lead, draft.body) == ("", "", "")


def test_finished_reasoning_is_stripped():
    draft = parse_article("<think>plan</think>\nHEADLINE: Lab opens\nLEAD: A lead.\nBODY: Body text.")
    assert draft.headline == "Lab opens"
    assert draft.body == "Body text."


# ------------------------------------------------------------ NLI evidence passages
def test_passages_split_long_descriptions_without_losing_text():
    sentences = [f"Sentence number {i} describes the scene in some detail." for i in range(100)]
    passages = FactualitySignal._passages([" ".join(sentences)], max_chars=500)
    assert len(passages) > 1
    assert all(len(p) <= 500 for p in passages)
    joined = " ".join(passages)
    assert all(s in joined for s in sentences)


# ------------------------------------------------------------ VideoLLaMA3 decoding
def test_generated_part_handles_both_generate_conventions():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from journalist_robot.models.videollama3_controller import generated_part

    prompt = torch.tensor([5, 6, 7])
    new = torch.tensor([1, 2])
    # inputs_embeds path (VideoLLaMA3): only new tokens are returned
    assert generated_part(new, prompt).tolist() == [1, 2]
    # input_ids path: prompt + new tokens
    assert generated_part(torch.cat([prompt, new]), prompt).tolist() == [1, 2]
