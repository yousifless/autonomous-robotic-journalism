"""Captured-view check: decide whether recorded footage is usable and on-topic."""

from __future__ import annotations

from dataclasses import dataclass

from journalist_robot.core.json_utils import parse_json_object

VALID_ACTIONS = ("NONE", "APPROACH", "HOLD", "ROTATE_LEFT", "ROTATE_RIGHT", "BACK_OFF")


@dataclass
class ViewCheckResult:
    view_usable: bool
    on_topic: bool
    suggested_action: str
    guidance: str

    @property
    def needs_feedback(self) -> bool:
        return not (self.view_usable and self.on_topic)

    def prompt_text(self) -> str:
        status = []
        if not self.view_usable:
            status.append("the subject is not clearly visible")
        if not self.on_topic:
            status.append("the view does not match the mission brief")
        reason = " and ".join(status) if status else "the view is acceptable"
        return f"Current footage: {reason}. Suggested action: {self.suggested_action}. {self.guidance}".strip()


def parse_view_check(text: str) -> ViewCheckResult:
    """Parse the checker's JSON answer; an unparseable answer is treated as usable.

    Treating parse failures as "usable" avoids steering the robot on the
    basis of a malformed answer.
    """
    obj = parse_json_object(text) or {}
    action = str(obj.get("suggested_action", "NONE")).upper().strip()
    if action not in VALID_ACTIONS:
        action = "NONE"
    guidance = str(obj.get("guidance", "")).strip()
    if guidance.lower() == "none":
        guidance = ""
    return ViewCheckResult(
        view_usable=_as_bool(obj.get("view_usable", True)),
        on_topic=_as_bool(obj.get("on_topic", True)),
        suggested_action=action,
        guidance=guidance,
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")
