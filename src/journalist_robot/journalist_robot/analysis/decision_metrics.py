"""Navigation metrics used in the paper's pilot (Sec. IV-B, Tables III and IV).

Commands are discretised into Left (omega > 0.2), Right (omega < -0.2),
Forward (v > 0) or Stop, and two decision sequences are compared index by
index over a fixed decision budget (70 in the paper):

* alignment  - fraction of indices with the same label as a reference
               (teleoperated) sequence;
* divergence - fraction of indices whose labels differ between two runs
               (e.g. the same environment under two mission briefs).

Matched indices need not correspond to matched scene states, so these are
behavioural-agreement proxies, not measures of navigation or reporting quality.

Command-line use on two ``decisions.jsonl`` files::

    ros2 run journalist_robot decision_metrics reference.jsonl candidate.jsonl --budget 70
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Sequence

LEFT, RIGHT, FORWARD, STOP = "Left", "Right", "Forward", "Stop"


def discretise(linear_x: float, angular_z: float,
               angular_threshold: float = 0.2,
               precedence: str = "rotation") -> str:
    """Map a velocity command to Left/Right/Forward/Stop.

    When a command both translates and rotates, ``precedence`` decides which
    label wins ("rotation" or "translation").
    """
    rotating = abs(angular_z) > angular_threshold
    forward = linear_x > 0.0
    rot_label = LEFT if angular_z > 0 else RIGHT
    if precedence == "rotation":
        if rotating:
            return rot_label
        return FORWARD if forward else STOP
    if forward:
        return FORWARD
    return rot_label if rotating else STOP


def labels_from_records(records: Iterable[Dict], **kwargs) -> List[str]:
    return [discretise(float(r["linear_x"]), float(r["angular_z"]), **kwargs) for r in records]


def alignment(reference: Sequence[str], candidate: Sequence[str], budget: int) -> float:
    n = min(budget, len(reference), len(candidate))
    if n == 0:
        return 0.0
    return sum(1 for a, b in zip(reference[:n], candidate[:n]) if a == b) / n


def divergence(run_a: Sequence[str], run_b: Sequence[str], budget: int) -> float:
    return 1.0 - alignment(run_a, run_b, budget)


def load_decisions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Index-matched alignment between two decision logs.")
    parser.add_argument("reference", help="decisions.jsonl of the reference run (e.g. teleoperation)")
    parser.add_argument("candidate", help="decisions.jsonl of the run to compare")
    parser.add_argument("--budget", type=int, default=70, help="number of decisions compared (default 70)")
    parser.add_argument("--precedence", choices=("rotation", "translation"), default="rotation")
    args = parser.parse_args(argv)

    ref = labels_from_records(load_decisions(args.reference), precedence=args.precedence)
    cand = labels_from_records(load_decisions(args.candidate), precedence=args.precedence)
    n = min(args.budget, len(ref), len(cand))
    print(f"decisions compared : {n}")
    print(f"alignment          : {100.0 * alignment(ref, cand, args.budget):.1f} %")
    print(f"divergence         : {100.0 * divergence(ref, cand, args.budget):.1f} %")


if __name__ == "__main__":
    main()
