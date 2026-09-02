"""Explicit stage gates and retry guidance for the science program."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_gate(
    experiment_id: str,
    *,
    evidence_complete: bool,
    thresholds_pass: bool,
    prerequisite_status: str = "pass",
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a stable decision without converting missing evidence to failure."""
    if not evidence_complete or prerequisite_status in {"blocked", "infrastructure_failure"}:
        status = "blocked"
    elif thresholds_pass:
        status = "pass"
    else:
        status = "scientific_fail"
    return {
        "experiment_id": experiment_id,
        "status": status,
        "evidence_complete": bool(evidence_complete),
        "thresholds_pass": bool(thresholds_pass),
        "prerequisite_status": prerequisite_status,
        "notes": list(notes),
        "retry_policy": {
            "blocked": "repair_inputs_or_environment",
            "scientific_fail": "research_and_targeted_fix",
            "pass": "advance",
        }[status],
    }


def retry_action(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status"))
    actions = {
        "blocked": "repair_inputs_or_environment",
        "scientific_fail": "research_and_targeted_fix",
        "pass": "advance",
    }
    if status not in actions:
        raise ValueError(f"unknown gate status: {status}")
    return actions[status]
