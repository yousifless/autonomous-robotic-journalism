"""Control policy of the perception & action loop.

The controller VLM proposes a velocity command together with a confidence and
a newsworthiness rating. This module turns that proposal into the command that
is actually sent to the base, and decides when recording starts and stops:

1. ``RecordingTrigger``  - hysteresis rule on N_t and confidence (Sec. III-A).
2. ``ControlPolicy``     - confidence gating, prompt-independent visual reflex
                           gating, turn cooldown, anti-stall exploration,
                           confidence/latency speed scaling, velocity limits.
3. ``ExplorationState``  - compact memory of recent behaviour that is fed back
                           into the prompt to discourage repetition.

All of this is pure Python so it can be unit tested without ROS or a GPU.

Note: the reflex gate is a lightweight image heuristic, not a geometric
collision-avoidance layer. Use a certified safety layer (bumpers, lidar
stop field, speed limiter on the base) for any deployment around people.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, List, Optional

FORWARD = "FORWARD"
BACKWARD = "BACKWARD"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
HOLD = "HOLD"

TRIGGER_START = "START"
TRIGGER_CONTINUE = "CONTINUE"
TRIGGER_STOP = "STOP"


@dataclass(frozen=True)
class VelocityLimits:
    max_linear: float = 0.4    # m/s  (paper, Sec. IV-B1)
    max_angular: float = 0.5   # rad/s


def clamp(value: float, low: float, high: float) -> float:
    """Clamp to [low, high]; a non-finite value becomes 0 (clamped)."""
    if not math.isfinite(value):
        value = 0.0
    return max(low, min(high, value))


def movement_label(linear_x: float, angular_z: float, deadband: float = 0.05) -> str:
    """Coarse label used for logging, telemetry and the prompt history."""
    if abs(linear_x) > deadband:
        return FORWARD if linear_x > 0 else BACKWARD
    if abs(angular_z) > deadband:
        return TURN_LEFT if angular_z > 0 else TURN_RIGHT
    return HOLD


# --------------------------------------------------------------------------
# Recording trigger
# --------------------------------------------------------------------------

@dataclass
class TriggerThresholds:
    start_newsworthiness: float = 0.7
    start_confidence: float = 0.8
    stop_newsworthiness: float = 0.3
    stop_confidence: float = 0.5
    max_segment_duration_s: float = 0.0   # 0 disables the time limit


class RecordingTrigger:
    """Hysteresis rule for starting and stopping a recording segment.

    * Start when ``N_t >= start_newsworthiness`` and ``c_t >= start_confidence``.
    * Stop when ``N_t < stop_newsworthiness`` and ``c_t < stop_confidence``,
      or when the optional maximum segment duration is exceeded.
    * Otherwise keep the current state (``CONTINUE``).
    """

    def __init__(self, thresholds: TriggerThresholds = TriggerThresholds()) -> None:
        self.thresholds = thresholds
        self.recording = False
        self._segment_start: Optional[float] = None

    def update(self, newsworthiness: float, confidence: float, now: float) -> str:
        t = self.thresholds
        if not self.recording:
            if newsworthiness >= t.start_newsworthiness and confidence >= t.start_confidence:
                self.recording = True
                self._segment_start = now
                return TRIGGER_START
            return TRIGGER_CONTINUE

        too_long = (
            t.max_segment_duration_s > 0.0
            and self._segment_start is not None
            and (now - self._segment_start) >= t.max_segment_duration_s
        )
        if too_long or (newsworthiness < t.stop_newsworthiness and confidence < t.stop_confidence):
            self.recording = False
            self._segment_start = None
            return TRIGGER_STOP
        return TRIGGER_CONTINUE

    def force_stop(self) -> bool:
        """Close an open segment (e.g. on shutdown). Returns True if one was open."""
        was_recording = self.recording
        self.recording = False
        self._segment_start = None
        return was_recording


# --------------------------------------------------------------------------
# Exploration memory
# --------------------------------------------------------------------------

@dataclass
class ExplorationState:
    phase: str = "SCAN"                 # SCAN | ADVANCE | FOLLOW
    last_action: str = HOLD
    consecutive_holds: int = 0
    stuck_counter: int = 0
    scan_direction: str = "LEFT"
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def update(self, label: str, newsworthiness: float, confidence: float) -> None:
        self.last_action = label
        self.history.append(label)

        if label == HOLD:
            self.consecutive_holds += 1
            if self.consecutive_holds >= 3:
                self.stuck_counter += 1
        else:
            self.consecutive_holds = 0

        if newsworthiness > 0.7 and confidence > 0.8:
            self.phase = "FOLLOW"
        elif label in (FORWARD, BACKWARD) and confidence > 0.6:
            self.phase = "ADVANCE"
        elif label in (TURN_LEFT, TURN_RIGHT):
            self.phase = "SCAN"
            self.scan_direction = "LEFT" if label == TURN_LEFT else "RIGHT"

        if label != HOLD and confidence > 0.5:
            self.stuck_counter = max(0, self.stuck_counter - 1)

    def recent_history(self, count: int = 5) -> str:
        recent: List[str] = list(self.history)[-count:]
        return " -> ".join(recent) if recent else "No previous commands"

    def prompt_summary(self) -> str:
        return (
            f"Phase={self.phase}, LastAction={self.last_action}, "
            f"ConsecutiveHolds={self.consecutive_holds}, ScanDirection={self.scan_direction}, "
            f"StuckCounter={self.stuck_counter}"
        )


# --------------------------------------------------------------------------
# Command policy
# --------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    limits: VelocityLimits = VelocityLimits()
    very_low_confidence: float = 0.2      # below: replace proposal by slow scanning
    low_confidence: float = 0.3           # below: halve forward speed
    turn_cooldown_s: float = 1.0          # min time before reversing turn direction
    hold_newsworthiness: float = 0.7      # holding position is allowed above this ...
    hold_confidence: float = 0.8          # ... and this confidence
    max_consecutive_holds: int = 3        # then force exploration
    slow_inference_s: float = 1.5         # above: scale speeds down
    slow_inference_scale: float = 0.6
    allow_reverse: bool = False           # the controller only sees the front camera


@dataclass
class PolicyInput:
    linear_x: float
    angular_z: float
    confidence: float
    newsworthiness: float
    reasoning: str = ""
    inference_latency_s: float = 0.0
    center_blocked: bool = False
    forward_consensus: bool = True
    preferred_turn: Optional[str] = None  # "LEFT" | "RIGHT" | None


@dataclass
class PolicyOutput:
    linear_x: float
    angular_z: float
    label: str
    notes: List[str]

    def reasoning(self, model_reasoning: str) -> str:
        if not self.notes:
            return model_reasoning
        return f"{model_reasoning} | Policy: {'; '.join(self.notes)}"


def confidence_speed_scale(confidence: float) -> float:
    if confidence >= 0.8:
        return 1.0
    if confidence >= 0.6:
        return 0.8
    if confidence >= 0.4:
        return 0.6
    return 0.4


class ControlPolicy:
    """Applies gating and limits to a VLM velocity proposal."""

    def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
        self.config = config
        self._last_turn_direction: Optional[str] = None
        self._last_turn_time = 0.0

    def apply(self, proposal: PolicyInput, state: ExplorationState, now: float) -> PolicyOutput:
        cfg = self.config
        lim = cfg.limits
        lx = clamp(proposal.linear_x, self._min_linear(), lim.max_linear)
        az = clamp(proposal.angular_z, -lim.max_angular, lim.max_angular)
        notes: List[str] = []
        conf = proposal.confidence

        # 1) Confidence gating.
        if conf < cfg.very_low_confidence:
            direction = 1.0 if state.scan_direction == "LEFT" else -1.0
            return self._finish(0.0, direction * 0.5 * lim.max_angular, conf, proposal.inference_latency_s,
                                ["very low confidence - slow scanning instead of the proposal"])
        if conf < cfg.low_confidence:
            lx *= 0.5
            notes.append("low confidence - forward speed halved")

        # 2) Reflex gate: never advance into a view that looks blocked.
        blocked = proposal.center_blocked or not proposal.forward_consensus
        if blocked and lx > 0.0:
            lx = 0.0
            notes.append("center view blocked or no forward consensus - forward suppressed")
            if abs(az) < 0.05 and proposal.preferred_turn:
                az = lim.max_angular if proposal.preferred_turn == "LEFT" else -lim.max_angular
                notes.append(f"reflex turn {proposal.preferred_turn}")

        # 3) Turn cooldown: avoid left/right oscillation.
        label = movement_label(lx, az)
        if label in (TURN_LEFT, TURN_RIGHT):
            direction = "LEFT" if label == TURN_LEFT else "RIGHT"
            if (
                self._last_turn_direction
                and direction != self._last_turn_direction
                and (now - self._last_turn_time) < cfg.turn_cooldown_s
            ):
                direction = self._last_turn_direction
                az = abs(az) if direction == "LEFT" else -abs(az)
                notes.append("turn cooldown - keeping previous direction")
            self._last_turn_direction = direction
            self._last_turn_time = now

        # 4) Anti-stall: holding is only justified for highly newsworthy views.
        if movement_label(lx, az) == HOLD:
            if proposal.newsworthiness >= cfg.hold_newsworthiness and conf >= cfg.hold_confidence:
                notes.append("high newsworthiness - holding position to observe")
            elif state.consecutive_holds >= cfg.max_consecutive_holds:
                if not blocked:
                    lx, az = 0.5 * lim.max_linear, 0.0
                    notes.append("repeated holds - forcing forward exploration")
                else:
                    direction = state.scan_direction
                    if state.consecutive_holds % 2 == 1:
                        direction = "RIGHT" if direction == "LEFT" else "LEFT"
                    lx, az = 0.0, (lim.max_angular if direction == "LEFT" else -lim.max_angular)
                    notes.append(f"repeated holds - forcing scan {direction}")

        return self._finish(lx, az, conf, proposal.inference_latency_s, notes)

    def _finish(self, lx: float, az: float, conf: float, latency: float, notes: List[str]) -> PolicyOutput:
        cfg = self.config
        scale = confidence_speed_scale(conf)
        if latency > cfg.slow_inference_s:
            scale *= cfg.slow_inference_scale
            notes.append(f"slow inference ({latency:.2f}s) - speeds scaled")
        lx = clamp(lx * scale, self._min_linear(), cfg.limits.max_linear)
        az = clamp(az * scale, -cfg.limits.max_angular, cfg.limits.max_angular)
        return PolicyOutput(linear_x=lx, angular_z=az, label=movement_label(lx, az), notes=notes)

    def _min_linear(self) -> float:
        """Reversing is blind (decisions use the front camera only), so it is off by default."""
        return -self.config.limits.max_linear if self.config.allow_reverse else 0.0
