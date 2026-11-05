import pytest

from journalist_robot.core.control_policy import (FORWARD, HOLD, TRIGGER_CONTINUE, TRIGGER_START, TRIGGER_STOP,
                                                  TURN_LEFT, TURN_RIGHT, ControlPolicy, ExplorationState,
                                                  PolicyConfig, PolicyInput, RecordingTrigger, TriggerThresholds,
                                                  VelocityLimits, movement_label)


def make_policy(max_linear=0.4, max_angular=0.5):
    return ControlPolicy(PolicyConfig(limits=VelocityLimits(max_linear, max_angular)))


# ---------------------------------------------------------------- trigger
def test_trigger_starts_only_above_both_thresholds():
    trig = RecordingTrigger()
    assert trig.update(0.75, 0.70, now=0.0) == TRIGGER_CONTINUE   # confidence too low
    assert trig.update(0.65, 0.95, now=1.0) == TRIGGER_CONTINUE   # newsworthiness too low
    assert trig.update(0.70, 0.80, now=2.0) == TRIGGER_START
    assert trig.recording


def test_trigger_hysteresis_requires_both_below_stop_thresholds():
    trig = RecordingTrigger()
    trig.update(0.9, 0.9, now=0.0)
    assert trig.update(0.2, 0.6, now=1.0) == TRIGGER_CONTINUE     # confidence still >= 0.5
    assert trig.update(0.5, 0.4, now=2.0) == TRIGGER_CONTINUE     # newsworthiness still >= 0.3
    assert trig.update(0.2, 0.4, now=3.0) == TRIGGER_STOP
    assert not trig.recording


def test_trigger_max_duration():
    trig = RecordingTrigger(TriggerThresholds(max_segment_duration_s=10.0))
    trig.update(0.9, 0.9, now=0.0)
    assert trig.update(0.9, 0.9, now=5.0) == TRIGGER_CONTINUE
    assert trig.update(0.9, 0.9, now=10.0) == TRIGGER_STOP


def test_force_stop():
    trig = RecordingTrigger()
    assert trig.force_stop() is False
    trig.update(0.9, 0.9, now=0.0)
    assert trig.force_stop() is True


# ----------------------------------------------------------------- policy
def test_velocities_are_clamped_to_limits():
    out = make_policy().apply(PolicyInput(2.0, 3.0, confidence=0.95, newsworthiness=0.5),
                              ExplorationState(), now=0.0)
    assert out.linear_x <= 0.4 + 1e-9
    assert abs(out.angular_z) <= 0.5 + 1e-9


def test_forward_suppressed_when_center_blocked():
    out = make_policy().apply(PolicyInput(0.3, 0.0, confidence=0.9, newsworthiness=0.5,
                                          center_blocked=True, preferred_turn="LEFT"),
                              ExplorationState(), now=0.0)
    assert out.linear_x == 0.0
    assert out.label == TURN_LEFT


def test_very_low_confidence_replaces_proposal_with_slow_scan():
    out = make_policy().apply(PolicyInput(0.4, 0.0, confidence=0.1, newsworthiness=0.9),
                              ExplorationState(), now=0.0)
    assert out.linear_x == 0.0
    assert out.label in (TURN_LEFT, TURN_RIGHT)
    assert abs(out.angular_z) < 0.5


def test_turn_cooldown_prevents_immediate_reversal():
    policy, state = make_policy(), ExplorationState()
    first = policy.apply(PolicyInput(0.0, 0.4, confidence=0.9, newsworthiness=0.3), state, now=0.0)
    second = policy.apply(PolicyInput(0.0, -0.4, confidence=0.9, newsworthiness=0.3), state, now=0.5)
    assert first.label == TURN_LEFT
    assert second.label == TURN_LEFT
    third = policy.apply(PolicyInput(0.0, -0.4, confidence=0.9, newsworthiness=0.3), state, now=2.0)
    assert third.label == TURN_RIGHT


def test_hold_allowed_for_highly_newsworthy_view():
    out = make_policy().apply(PolicyInput(0.0, 0.0, confidence=0.9, newsworthiness=0.8),
                              ExplorationState(consecutive_holds=5), now=0.0)
    assert out.label == HOLD


def test_repeated_holds_force_exploration():
    out = make_policy().apply(PolicyInput(0.0, 0.0, confidence=0.9, newsworthiness=0.2),
                              ExplorationState(consecutive_holds=3), now=0.0)
    assert out.label == FORWARD


def test_confidence_scaling():
    high = make_policy().apply(PolicyInput(0.4, 0.0, 0.9, 0.5), ExplorationState(), now=0.0)
    mid = make_policy().apply(PolicyInput(0.4, 0.0, 0.65, 0.5), ExplorationState(), now=0.0)
    assert high.linear_x == pytest.approx(0.4)
    assert mid.linear_x == pytest.approx(0.32)


def test_movement_label():
    assert movement_label(0.2, 0.0) == FORWARD
    assert movement_label(0.0, 0.3) == TURN_LEFT
    assert movement_label(0.0, -0.3) == TURN_RIGHT
    assert movement_label(0.0, 0.0) == HOLD


def test_exploration_state_history():
    s = ExplorationState()
    for label in (FORWARD, FORWARD, TURN_LEFT):
        s.update(label, 0.3, 0.7)
    assert s.recent_history(2) == "FORWARD -> TURN_LEFT"
    assert s.scan_direction == "LEFT"
