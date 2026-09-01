"""Rotation-stable geometric pocket openness from ray-sphere intersections."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


RADIUS_TABLE_VERSION = "bondi-1964-v1"
UNKNOWN_RADIUS_ANGSTROM = 1.70
VDW_RADII_ANGSTROM: Mapping[str, float] = {
    "H": 1.20,
    "HE": 1.40,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "NE": 1.54,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "AR": 1.88,
    "AS": 1.85,
    "SE": 1.90,
    "BR": 1.85,
    "KR": 2.02,
    "TE": 2.06,
    "I": 1.98,
    "XE": 2.16,
}


@dataclass(frozen=True)
class ParsedAtoms:
    coordinates: np.ndarray
    elements: tuple[str, ...]
    unknown_element_count: int


@dataclass(frozen=True)
class OpennessResult:
    blocked_rays: int
    direction_count: int
    enclosure: float
    openness: float
    cutoff_angstrom: float
    nearest_distance: float | None
    atom_density: float
    heavy_atom_count: int
    unknown_element_count: int
    radius_table_version: str = RADIUS_TABLE_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fibonacci_directions(count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("direction count must be positive")
    index = np.arange(count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * index
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def ray_sphere_blocked(
    directions: np.ndarray,
    relative_centers: np.ndarray,
    radii: np.ndarray,
    *,
    cutoff: float,
    atom_chunk_size: int = 4096,
) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    relative_centers = np.asarray(relative_centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3)")
    if relative_centers.ndim != 2 or relative_centers.shape[1] != 3:
        raise ValueError("relative atom centers must have shape (M, 3)")
    if len(relative_centers) != len(radii):
        raise ValueError("one radius is required per atom")
    if cutoff <= 0 or atom_chunk_size <= 0:
        raise ValueError("cutoff and atom chunk size must be positive")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("ray directions must be unit length")

    blocked = np.zeros(len(directions), dtype=bool)
    for start in range(0, len(relative_centers), atom_chunk_size):
        if blocked.all():
            break
        centers = relative_centers[start : start + atom_chunk_size]
        chunk_radii = radii[start : start + atom_chunk_size]
        # Projection b and perpendicular-distance term q for
        # ||t*d - p||^2 <= r^2, with ray origin at zero.
        projection = directions @ centers.T
        q = np.einsum("ij,ij->i", centers, centers) - chunk_radii * chunk_radii
        discriminant = projection * projection - q[None, :]
        intersects_line = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant, 0.0))
        entry = projection - root
        exit_ = projection + root
        intersects_ray_segment = intersects_line & (exit_ >= 0.0) & (entry <= cutoff)
        blocked |= intersects_ray_segment.any(axis=1)
    return blocked


def _normalize_element(value: str) -> str:
    return value.strip().upper()


def compute_openness(
    coordinates: np.ndarray,
    elements: Iterable[str],
    center: np.ndarray,
    *,
    direction_count: int = 2048,
    cutoff_angstrom: float = 12.0,
    radius_table: Mapping[str, float] = VDW_RADII_ANGSTROM,
    radius_table_version: str = RADIUS_TABLE_VERSION,
) -> OpennessResult:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    elements = tuple(_normalize_element(element) for element in elements)
    if coordinates.shape != (len(elements), 3):
        raise ValueError("coordinates and elements must describe the same atoms")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    relative = coordinates - center
    unknown_count = sum(element not in radius_table for element in elements)
    radii = np.array(
        [radius_table.get(element, UNKNOWN_RADIUS_ANGSTROM) for element in elements],
        dtype=np.float64,
    )
    directions = fibonacci_directions(direction_count)
    blocked = ray_sphere_blocked(
        directions, relative, radii, cutoff=cutoff_angstrom
    )
    blocked_count = int(blocked.sum())
    enclosure = blocked_count / direction_count
    distances = np.linalg.norm(relative, axis=1)
    sphere_volume = (4.0 / 3.0) * np.pi * cutoff_angstrom**3
    within_cutoff = int(np.count_nonzero(distances <= cutoff_angstrom))
    return OpennessResult(
        blocked_rays=blocked_count,
        direction_count=direction_count,
        enclosure=enclosure,
        openness=1.0 - enclosure,
        cutoff_angstrom=cutoff_angstrom,
        nearest_distance=float(distances.min()) if len(distances) else None,
        atom_density=within_cutoff / sphere_volume,
        heavy_atom_count=len(elements),
        unknown_element_count=unknown_count,
        radius_table_version=radius_table_version,
    )


def _infer_element(line: str) -> str:
    explicit = line[76:78].strip() if len(line) >= 78 else ""
    if explicit:
        return _normalize_element(explicit)
    atom_name = line[12:16].strip() if len(line) >= 16 else ""
    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        return "UNKNOWN"
    two_letter = letters[:2].upper()
    return two_letter if two_letter in VDW_RADII_ANGSTROM else letters[0].upper()


def parse_pdb_heavy_atoms(path: Path) -> ParsedAtoms:
    coordinates: list[tuple[float, float, float]] = []
    elements: list[str] = []
    unknown_count = 0
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii", errors="replace").splitlines(), start=1
    ):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = _infer_element(line)
        if element == "H":
            continue
        try:
            coordinate = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError as exc:
            raise ValueError(f"invalid PDB coordinates at {path}:{line_number}") from exc
        if element not in VDW_RADII_ANGSTROM:
            unknown_count += 1
        coordinates.append(coordinate)
        elements.append(element)
    array = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
    return ParsedAtoms(array, tuple(elements), unknown_count)


def select_smoke_records(
    records: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: (float(record["openness"]), str(record["pocket_id"])),
    )
    if len(ordered) < 6:
        raise ValueError("at least six pockets are required for smoke selection")
    if len(ordered) == 30:
        smoke_indices = (0, 1, 14, 15, 28, 29)
    else:
        middle_left = (len(ordered) - 1) // 2
        middle_right = len(ordered) // 2
        smoke_indices = (0, 1, middle_left, middle_right, len(ordered) - 2, len(ordered) - 1)
    smoke = [ordered[index] for index in smoke_indices]

    boundaries = (len(ordered) // 3, 2 * len(ordered) // 3)
    groups = (
        ("closed", ordered[: boundaries[0]]),
        ("middle", ordered[boundaries[0] : boundaries[1]]),
        ("open", ordered[boundaries[1] :]),
    )
    enriched: list[dict[str, object]] = []
    for label, group in groups:
        for index, record in enumerate(group):
            updated = dict(record)
            updated["openness_tertile"] = label
            updated["wrong_pocket_id"] = group[(index + 1) % len(group)]["pocket_id"]
            enriched.append(updated)
    return smoke, sorted(enriched, key=lambda record: str(record["pocket_id"]))
