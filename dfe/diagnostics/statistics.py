"""Pocket-clustered effects, bootstrap intervals and FDR for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class InteractionFit:
    intervention: str
    interaction: float
    coefficients: Mapping[str, float]


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    draws: int
    seed: int


def paired_effect(
    baseline: Mapping[str, float], intervention: Mapping[str, float]
) -> float:
    keys = sorted(set(baseline) & set(intervention))
    if not keys:
        raise ValueError("paired effect requires at least one shared pocket")
    return float(np.mean([intervention[key] - baseline[key] for key in keys]))


def benjamini_hochberg(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=np.float64)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be between zero and one")
    if not len(values):
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def sample_pocket_clusters(
    rows: Iterable[Mapping[str, object]], generator: np.random.Generator
) -> list[dict[str, object]]:
    records = [dict(row) for row in rows]
    by_pocket: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_pocket.setdefault(str(record["pocket_id"]), []).append(record)
    pocket_ids = sorted(by_pocket)
    if not pocket_ids:
        raise ValueError("cluster bootstrap requires pockets")
    draws = generator.choice(pocket_ids, size=len(pocket_ids), replace=True)
    sampled: list[dict[str, object]] = []
    for draw_index, pocket_id in enumerate(draws):
        for record in by_pocket[str(pocket_id)]:
            value = dict(record)
            value["cluster_draw_id"] = draw_index
            sampled.append(value)
    return sampled


def _design_matrix(
    rows: Iterable[Mapping[str, object]], intervention: str
) -> tuple[np.ndarray, np.ndarray]:
    records = [dict(row) for row in rows]
    filtered = [
        record
        for record in records
        if record["intervention"] in {"D0", intervention}
    ]
    if not filtered:
        raise ValueError("no baseline/intervention rows for regression")
    columns = []
    target = []
    for record in filtered:
        arm = 1.0 if record["intervention"] == intervention else 0.0
        openness = float(record["openness"])
        columns.append(
            [
                1.0,
                arm,
                openness,
                arm * openness,
                float(record["pocket_atom_count"]),
                float(record["reference_ligand_heavy_atoms"]),
            ]
        )
        target.append(float(record["metric"]))
    return np.asarray(columns, dtype=np.float64), np.asarray(target, dtype=np.float64)


def fit_openness_interaction(
    rows: Iterable[Mapping[str, object]], *, intervention: str
) -> InteractionFit:
    matrix, target = _design_matrix(rows, intervention)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    names = (
        "intercept",
        "intervention",
        "openness",
        "intervention:openness",
        "pocket_atom_count",
        "reference_ligand_heavy_atoms",
    )
    values = {name: float(value) for name, value in zip(names, coefficients)}
    return InteractionFit(intervention, values["intervention:openness"], values)


def cluster_bootstrap_ci(
    rows: Iterable[Mapping[str, object]],
    statistic: Callable[[list[dict[str, object]]], float],
    *,
    draws: int = 10000,
    seed: int = 20260901,
) -> BootstrapCI:
    records = [dict(row) for row in rows]
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    generator = np.random.Generator(np.random.PCG64(seed))
    estimate = float(statistic(records))
    values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        values[index] = statistic(sample_pocket_clusters(records, generator))
    return BootstrapCI(
        estimate=estimate,
        lower=float(np.quantile(values, 0.025)),
        upper=float(np.quantile(values, 0.975)),
        draws=draws,
        seed=seed,
    )


def smoke_gate(
    evidence: Mapping[str, object], *, retained_arm_ids: Iterable[str]
) -> dict[str, object]:
    checks = {
        "all_jobs_terminal": evidence.get("terminal_job_count")
        == evidence.get("expected_job_count"),
        "exact_attempt_counts": evidence.get("jobs_with_exact_attempt_count")
        == evidence.get("expected_job_count"),
        "ten_attempts_per_job": evidence.get("expected_attempts_per_job") == 10,
        "normal_parity": evidence.get("normal_parity_passed") is True,
        "finite_traces": evidence.get("finite_traces") is True,
        "clean_replay": evidence.get("clean_replay") is True,
        "output_hashes": evidence.get("output_hashes_valid") is True,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "evidence": dict(evidence),
        "retained_arm_ids": list(retained_arm_ids),
    }
