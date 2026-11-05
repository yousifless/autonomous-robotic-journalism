"""Robust extraction of the controller's JSON answer from free-form VLM output."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional


def extract_balanced_json(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` object in ``text``, or ``None``.

    Braces inside JSON strings are ignored, so reasoning text that contains
    ``{`` or ``}`` does not break the match.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _match_float(pattern: str, text: str, default: float) -> float:
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def _match_reasoning(text: str) -> str:
    match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match:
        return match.group(1).strip()
    return "Model reasoning not available"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _unit(value: Any) -> float:
    """Score in [0, 1]. Answers on a 0-100 scale are rescaled; non-finite, missing or larger values give 0."""
    value = _finite(value, 0.0)
    if 1.0 < value <= 100.0:
        value /= 100.0
    elif value > 100.0:
        value = 0.0
    return min(1.0, max(0.0, value))


def _sanitize(cmd: Dict[str, Any]) -> Dict[str, Any]:
    cmd["linear_x"] = _finite(cmd["linear_x"])
    cmd["angular_z"] = _finite(cmd["angular_z"])
    cmd["confidence"] = _unit(cmd["confidence"])
    cmd["newsworthiness_score"] = _unit(cmd["newsworthiness_score"])
    return cmd


def parse_controller_output(text: str) -> Dict[str, Any]:
    """Parse the controller answer into a command dictionary.

    The VLM is asked for a single JSON object. If the object is malformed or
    truncated, individual fields are recovered with regular expressions and
    the result is marked with ``"parsed_with_fallback": True``. Missing or
    non-finite velocities become zero (no motion); confidence and
    newsworthiness are clipped to [0, 1] and default to zero, so a garbled
    answer can never produce a confident motion command.
    """
    extracted = extract_balanced_json(text)
    if extracted:
        try:
            obj = json.loads(extracted)
            if isinstance(obj, dict):
                return _sanitize({
                    "linear_x": obj.get("linear_x", 0.0),
                    "angular_z": obj.get("angular_z", 0.0),
                    "confidence": obj.get("confidence", 0.0),
                    "newsworthiness_score": obj.get("newsworthiness_score", 0.0),
                    "reasoning": str(obj.get("reasoning", "")),
                    "parsed_with_fallback": False,
                })
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return _sanitize({
        "linear_x": _match_float(r'"linear_x"\s*:\s*([-0-9.]+)', text, 0.0),
        "angular_z": _match_float(r'"angular_z"\s*:\s*([-0-9.]+)', text, 0.0),
        "confidence": _match_float(r'"confidence"\s*:\s*([-0-9.]+)', text, 0.0),
        "newsworthiness_score": _match_float(r'"newsworthiness_score"\s*:\s*([-0-9.]+)', text, 0.0),
        "reasoning": _match_reasoning(text),
        "parsed_with_fallback": True,
    })


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object in ``text`` as a dict, or ``None``."""
    extracted = extract_balanced_json(text)
    if not extracted:
        return None
    try:
        obj = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
