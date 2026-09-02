"""Science-facing SE(3) comparisons built on the Phase 0 primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dfe.diagnostics.se3 import ErrorStats, normalized_error


def compare_invariant(reference: np.ndarray, transformed: np.ndarray) -> ErrorStats:
    return normalized_error(np.asarray(transformed), np.asarray(reference))


def compare_vector(reference: np.ndarray, transformed: np.ndarray, rotation: np.ndarray) -> ErrorStats:
    expected = np.asarray(reference) @ np.asarray(rotation).T
    return normalized_error(np.asarray(transformed), expected)


def compare_position(
    reference: np.ndarray,
    transformed: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> ErrorStats:
    expected = np.asarray(reference) @ np.asarray(rotation).T + np.asarray(translation)
    return normalized_error(np.asarray(transformed), expected)


def first_discrete_divergence(reference: Sequence[int], transformed: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, transformed)):
        if left != right:
            return index
    if len(reference) != len(transformed):
        return min(len(reference), len(transformed))
    return None


def summarize_errors(errors: Sequence[float], tolerance: float) -> dict[str, object]:
    values = np.asarray(tuple(errors), dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one error is required")
    if not np.isfinite(values).all():
        raise ValueError("errors must be finite")
    return {
        "count": int(values.size),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "tolerance": float(tolerance),
        "passed": bool(np.max(values) < tolerance),
    }
