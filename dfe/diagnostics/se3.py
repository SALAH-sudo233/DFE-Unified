"""Deterministic SE(3) transforms and reusable numerical comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorStats:
    normalized_max: float
    normalized_median: float
    normalized_p95: float
    absolute_max: float
    absolute_median: float
    absolute_p95: float


def sample_so3(seed: int, count: int) -> np.ndarray:
    """Sample deterministic Haar-distributed proper rotations from quaternions."""
    if count <= 0:
        raise ValueError("rotation count must be positive")
    generator = np.random.Generator(np.random.PCG64(seed))
    quaternion = generator.normal(size=(count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    rotations = np.empty((count, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def _validate_rotation(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation)
    if value.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return value


def apply_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points)
    rotation = _validate_rotation(rotation)
    if points.shape[-1:] != (3,):
        raise ValueError("points must have a final xyz dimension")
    moved = points @ rotation.T
    if translation is not None:
        translation = np.asarray(translation)
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        moved = moved + translation
    return moved


def apply_vectors(
    vectors: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    del translation
    vectors = np.asarray(vectors)
    rotation = _validate_rotation(rotation)
    if vectors.shape[-1:] != (3,):
        raise ValueError("vectors must have a final xyz dimension")
    return vectors @ rotation.T


def normalized_error(
    actual: np.ndarray, expected: np.ndarray, eps: float = 1e-12
) -> ErrorStats:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if actual.ndim == 0:
        delta = np.abs(actual - expected).reshape(1)
        scale = np.maximum(np.abs(expected), eps).reshape(1)
    elif actual.shape[-1:] == (3,):
        delta = np.linalg.norm(actual - expected, axis=-1).reshape(-1)
        scale = np.maximum(np.linalg.norm(expected, axis=-1), eps).reshape(-1)
    else:
        delta = np.abs(actual - expected).reshape(-1)
        scale = np.maximum(np.abs(expected), eps).reshape(-1)
    if delta.size == 0:
        return ErrorStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    relative = delta / scale
    return ErrorStats(
        normalized_max=float(np.max(relative)),
        normalized_median=float(np.median(relative)),
        normalized_p95=float(np.quantile(relative, 0.95)),
        absolute_max=float(np.max(delta)),
        absolute_median=float(np.median(delta)),
        absolute_p95=float(np.quantile(delta, 0.95)),
    )


def compare_invariant(actual: np.ndarray, expected: np.ndarray) -> ErrorStats:
    return normalized_error(actual, expected)


def compare_equivariant(
    transformed: np.ndarray,
    reference: np.ndarray,
    rotation: np.ndarray,
) -> ErrorStats:
    """Inverse-align a transformed row-vector output before comparison."""
    rotation = _validate_rotation(rotation)
    aligned = apply_vectors(transformed, rotation.T)
    return normalized_error(aligned, reference)
