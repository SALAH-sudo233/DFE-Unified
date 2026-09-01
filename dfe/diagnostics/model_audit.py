"""Transformation-law comparisons for named Pocket2Mol tensor events."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from .observer import TensorObserver
from .se3 import ErrorStats, apply_points, compare_equivariant, compare_invariant


EVENT_LAWS = {
    "df.raw": "mixed_df",
    "df.hidden": "invariant",
    "df.projected": "invariant",
    "encoder.scalar": "invariant",
    "encoder.vector": "equivariant",
    "frontier.logits": "invariant",
    "frontier.indices": "exact",
    "position.relative_mu": "equivariant",
    "position.absolute_mu": "point",
    "position.sigma": "invariant",
    "position.pi": "invariant",
    "element.logits": "invariant",
    "element.probability": "invariant",
    "bond.logits": "invariant",
    "bond.probability": "invariant",
    "termination.has_frontier": "exact",
}


@dataclass(frozen=True)
class EventAudit:
    key: str
    law: str
    passed: bool
    normalized_max: float | None = None
    normalized_median: float | None = None
    normalized_p95: float | None = None
    absolute_max: float | None = None
    absolute_median: float | None = None
    absolute_p95: float | None = None
    mismatch_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    first_failure: str | None
    events: tuple[EventAudit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "first_failure": self.first_failure,
            "events": [event.to_dict() for event in self.events],
        }


def _numeric_audit(
    key: str,
    law: str,
    stats: ErrorStats,
    tolerance: float,
) -> EventAudit:
    return EventAudit(
        key=key,
        law=law,
        passed=stats.normalized_max < tolerance,
        normalized_max=stats.normalized_max,
        normalized_median=stats.normalized_median,
        normalized_p95=stats.normalized_p95,
        absolute_max=stats.absolute_max,
        absolute_median=stats.absolute_median,
        absolute_p95=stats.absolute_p95,
    )


def _compare_mixed_df(
    actual: np.ndarray,
    expected: np.ndarray,
    rotation: np.ndarray,
) -> ErrorStats:
    if actual.shape[-1] != 8 or expected.shape[-1] != 8:
        raise ValueError("df.raw must have exactly eight features")
    aligned = actual.copy()
    aligned[..., 1:4] = actual[..., 1:4] @ rotation
    return compare_invariant(aligned, expected)


def compare_event_sets(
    reference: TensorObserver,
    transformed: TensorObserver,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    tolerance: float,
) -> AuditReport:
    reference_keys = reference.keys()
    transformed_keys = transformed.keys()
    transformed_key_set = set(transformed_keys)

    events: list[EventAudit] = []
    first_failure = None
    for step, event in reference_keys:
        if event not in EVENT_LAWS:
            raise ValueError(f"no transformation law declared for {event}")
        key = f"{step}:{event}"
        if (step, event) not in transformed_key_set:
            audit = EventAudit(
                key=key,
                law="missing_event",
                passed=False,
                mismatch_count=1,
            )
            events.append(audit)
            if first_failure is None:
                first_failure = key
            continue
        law = EVENT_LAWS[event]
        expected = reference.get(step, event).numpy()
        actual = transformed.get(step, event).numpy()
        if law == "exact":
            mismatch_count = (
                int(np.count_nonzero(actual != expected))
                if actual.shape == expected.shape
                else max(actual.size, expected.size)
            )
            audit = EventAudit(
                key=key,
                law=law,
                passed=actual.shape == expected.shape and mismatch_count == 0,
                mismatch_count=mismatch_count,
            )
        elif law == "equivariant":
            audit = _numeric_audit(
                key,
                law,
                compare_equivariant(actual, expected, rotation),
                tolerance,
            )
        elif law == "point":
            expected_transformed = apply_points(expected, rotation, translation)
            audit = _numeric_audit(
                key,
                law,
                compare_invariant(actual, expected_transformed),
                tolerance,
            )
        elif law == "mixed_df":
            audit = _numeric_audit(
                key,
                law,
                _compare_mixed_df(actual, expected, rotation),
                tolerance,
            )
        else:
            audit = _numeric_audit(
                key,
                law,
                compare_invariant(actual, expected),
                tolerance,
            )
        events.append(audit)
        if not audit.passed and first_failure is None:
            first_failure = key
    reference_key_set = set(reference_keys)
    for step, event in transformed_keys:
        if (step, event) in reference_key_set:
            continue
        key = f"{step}:{event}"
        events.append(
            EventAudit(
                key=key,
                law="extra_event",
                passed=False,
                mismatch_count=1,
            )
        )
        if first_failure is None:
            first_failure = key
    return AuditReport(first_failure is None, first_failure, tuple(events))


def audit_analytical_df_state(
    module,
    query_points: Tensor,
    pocket_pos: Tensor,
    pocket_types: Tensor,
    pocket_mask: Tensor,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    tolerance: float,
) -> AuditReport:
    dtype = query_points.dtype
    device = query_points.device
    rotation_tensor = torch.as_tensor(rotation, dtype=dtype, device=device)
    translation_tensor = torch.as_tensor(translation, dtype=dtype, device=device)
    transformed_query = query_points @ rotation_tensor.T + translation_tensor
    transformed_pocket = pocket_pos @ rotation_tensor.T + translation_tensor
    with torch.no_grad():
        reference = module.raw_features(
            query_points, pocket_pos, pocket_types, pocket_mask
        )
        transformed = module.raw_features(
            transformed_query, transformed_pocket, pocket_types, pocket_mask
        )
    stats = _compare_mixed_df(
        transformed.detach().cpu().numpy(),
        reference.detach().cpu().numpy(),
        np.asarray(rotation),
    )
    audit = _numeric_audit("analytical:df.raw", "mixed_df", stats, tolerance)
    return AuditReport(audit.passed, None if audit.passed else audit.key, (audit,))
