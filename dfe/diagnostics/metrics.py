"""End-to-end generation metrics computed from requested-attempt denominators."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


METRIC_DEFINITION_VERSION = "phase0-metrics.v1"
FAILURE_CODES = {
    "init_no_frontier",
    "init_threshold_exhausted",
    "early_no_frontier",
    "max_steps",
    "queue_empty",
    "reconstruction_error",
    "disconnected",
    "sanitize_error",
    "sdf_write_error",
    "docking_error",
    "posebusters_error",
}


def aggregation_key(record: Mapping[str, object]) -> tuple[object, object, object]:
    return record["pocket_id"], record["seed"], record["arm_id"]


def _rate(numerator: int, denominator: int, *, computable: int | None = None):
    value: dict[str, object] = {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }
    if computable is not None:
        value["computable"] = computable
    return value


def summarize_attempts(attempts: Iterable[Mapping[str, object]]) -> dict[str, object]:
    records = [dict(record) for record in attempts]
    attempt_ids = [record.get("attempt_id") for record in records]
    if any(not isinstance(value, str) or not value for value in attempt_ids):
        raise ValueError("every attempt requires a non-empty attempt_id")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("attempt IDs must be unique terminal records")
    denominator = len(records)
    valid = [
        record
        for record in records
        if record.get("status") == "evaluated" and bool(record.get("smiles"))
    ]
    docked = [record for record in records if record.get("docking_score") is not None]
    pose_computable = [record for record in records if record.get("posebusters_pass") is not None]
    pose_passed = [record for record in pose_computable if record["posebusters_pass"] is True]
    unique_smiles = {str(record["smiles"]) for record in valid}
    taxonomy: Counter[str] = Counter()
    for record in records:
        if record in valid:
            taxonomy["success"] += 1
            continue
        code = record.get("error_code")
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code: {code}")
        taxonomy[str(code)] += 1
    return {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "attempt_count": denominator,
        "validity": _rate(len(valid), denominator),
        "dockable": _rate(len(docked), denominator, computable=len(docked)),
        "posebusters": _rate(
            len(pose_passed), denominator, computable=len(pose_computable)
        ),
        "uniqueness": _rate(len(unique_smiles), len(valid)),
        "failure_taxonomy": dict(sorted(taxonomy.items())),
    }


def terminal_attempt_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        attempt_id = str(record["attempt_id"])
        latest[attempt_id] = dict(record)
    nonterminal = sorted(
        attempt_id
        for attempt_id, record in latest.items()
        if record.get("status") not in {"evaluated", "failed"}
    )
    if nonterminal:
        raise ValueError(f"attempts are not terminal: {nonterminal[:5]}")
    return [latest[key] for key in sorted(latest)]
