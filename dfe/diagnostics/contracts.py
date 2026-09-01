"""Immutable contracts and frozen configuration for Phase 0 diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PHASE0_CHECKPOINT_SHA256 = (
    "34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0"
)
PHASE0_SEEDS = (20260901, 20260902, 20260903)
PHASE0_INTERVENTIONS = ("D0", "D1", "D2", "D3", "D4", "D5")


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_new_manifest(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value))


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    run_id: str
    pocket_id: str
    seed: int
    intervention: str
    sample_index: int
    status: str = "requested"
    sampling_status: str = "pending"
    reconstruction_status: str = "pending"
    evaluation_status: str = "pending"
    error_code: str | None = None

    @classmethod
    def new(
        cls,
        run_id: str,
        pocket_id: str,
        seed: int,
        intervention: str,
        sample_index: int,
    ) -> "AttemptRecord":
        attempt_id = (
            f"{run_id}:{pocket_id}:{seed}:{intervention}:{sample_index:04d}"
        )
        return cls(
            attempt_id,
            run_id,
            pocket_id,
            seed,
            intervention,
            sample_index,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PocketRecord:
    pocket_id: str
    center: Mapping[str, float]
    inputs: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not self.pocket_id.strip():
            raise ValueError("pocket_id must not be empty")
        if set(self.center) != {"x", "y", "z"}:
            raise ValueError("pocket center must contain exactly x, y and z")


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    stage: str
    pocket_id: str
    seed: int
    intervention: str
    attempt_count: int

    def __post_init__(self) -> None:
        if self.stage not in {"smoke", "main"}:
            raise ValueError("job stage must be smoke or main")
        expected = 10 if self.stage == "smoke" else 20
        if self.attempt_count != expected:
            raise ValueError(
                f"{self.stage} attempt count must be exactly {expected}"
            )
        allowed_seeds = PHASE0_SEEDS[:1] if self.stage == "smoke" else PHASE0_SEEDS
        if self.seed not in allowed_seeds:
            raise ValueError("job seed must be one of the frozen Phase 0 seeds")
        if self.intervention not in PHASE0_INTERVENTIONS:
            raise ValueError("job intervention is not in the frozen Phase 0 matrix")

    @property
    def key(self) -> tuple[str, str, int, str]:
        return self.stage, self.pocket_id, self.seed, self.intervention


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    checkpoint_sha256: str
    pockets: tuple[PocketRecord, ...]
    jobs: tuple[JobSpec, ...]
    schema_version: str = "phase0.v1"

    @classmethod
    def new(
        cls,
        run_id: str,
        checkpoint_sha256: str,
        pockets: tuple[PocketRecord, ...],
        jobs: tuple[JobSpec, ...],
    ) -> "RunManifest":
        manifest = cls(run_id, checkpoint_sha256, tuple(pockets), tuple(jobs))
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != "phase0.v1":
            raise ValueError("manifest schema must be phase0.v1")
        if self.checkpoint_sha256 != PHASE0_CHECKPOINT_SHA256:
            raise ValueError("checkpoint hash does not match the frozen DF 500K anchor")
        pocket_ids = [record.pocket_id for record in self.pockets]
        if len(pocket_ids) != len(set(pocket_ids)):
            raise ValueError("duplicate pocket IDs are not allowed")
        job_keys = [job.key for job in self.jobs]
        if len(job_keys) != len(set(job_keys)):
            raise ValueError("duplicate job keys are not allowed")
        unknown_pockets = sorted({job.pocket_id for job in self.jobs} - set(pocket_ids))
        if unknown_pockets:
            raise ValueError(f"jobs reference unknown pockets: {unknown_pockets}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointConfig:
    path: Path
    sha256: str
    iteration: int
    state_tensor_count: int


@dataclass(frozen=True)
class OpennessConfig:
    direction_count: int
    cutoff_angstrom: float
    radius_table_version: str


@dataclass(frozen=True)
class SE3Config:
    rotations: int
    translations: int
    model_float32_tolerance: float
    analytical_float64_tolerance: float


@dataclass(frozen=True)
class Phase0Config:
    schema_version: str
    experiment_ids: Mapping[str, str]
    seeds: tuple[int, ...]
    smoke_attempts: int
    main_attempts: int
    checkpoint: CheckpointConfig
    sampling_policy: Path
    openness: OpennessConfig
    interventions: tuple[str, ...]
    d5_gates: tuple[float, ...]
    se3: SE3Config


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown {label} keys: {unknown}")
    if missing:
        raise ValueError(f"missing {label} keys: {missing}")


def load_phase0_config(path: Path) -> Phase0Config:
    path = Path(path).resolve()
    raw = _require_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), "config"
    )
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "experiment_ids",
            "seeds",
            "smoke_attempts",
            "main_attempts",
            "checkpoint",
            "sampling_policy",
            "openness",
            "interventions",
            "d5_gates",
            "se3",
        },
        "config",
    )

    experiment_ids = _require_mapping(raw["experiment_ids"], "experiment_ids")
    _require_exact_keys(
        experiment_ids,
        {"manifest", "openness", "se3", "smoke", "main", "trace", "statistics"},
        "experiment_ids",
    )
    checkpoint = _require_mapping(raw["checkpoint"], "checkpoint")
    _require_exact_keys(
        checkpoint, {"path", "sha256", "iteration", "state_tensor_count"}, "checkpoint"
    )
    openness = _require_mapping(raw["openness"], "openness")
    _require_exact_keys(
        openness,
        {"direction_count", "cutoff_angstrom", "radius_table_version"},
        "openness",
    )
    se3 = _require_mapping(raw["se3"], "se3")
    _require_exact_keys(
        se3,
        {
            "rotations",
            "translations",
            "model_float32_tolerance",
            "analytical_float64_tolerance",
        },
        "se3",
    )

    config = Phase0Config(
        schema_version=str(raw["schema_version"]),
        experiment_ids=dict(experiment_ids),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        smoke_attempts=int(raw["smoke_attempts"]),
        main_attempts=int(raw["main_attempts"]),
        checkpoint=CheckpointConfig(
            path=(path.parent / str(checkpoint["path"])).resolve(),
            sha256=str(checkpoint["sha256"]),
            iteration=int(checkpoint["iteration"]),
            state_tensor_count=int(checkpoint["state_tensor_count"]),
        ),
        sampling_policy=(path.parent / str(raw["sampling_policy"])).resolve(),
        openness=OpennessConfig(
            direction_count=int(openness["direction_count"]),
            cutoff_angstrom=float(openness["cutoff_angstrom"]),
            radius_table_version=str(openness["radius_table_version"]),
        ),
        interventions=tuple(str(item) for item in raw["interventions"]),
        d5_gates=tuple(float(item) for item in raw["d5_gates"]),
        se3=SE3Config(
            rotations=int(se3["rotations"]),
            translations=int(se3["translations"]),
            model_float32_tolerance=float(se3["model_float32_tolerance"]),
            analytical_float64_tolerance=float(se3["analytical_float64_tolerance"]),
        ),
    )
    _validate_phase0_config(config)
    return config


def _validate_phase0_config(config: Phase0Config) -> None:
    if config.schema_version != "phase0.v1":
        raise ValueError("schema_version must be phase0.v1")
    if config.seeds != PHASE0_SEEDS:
        raise ValueError("seeds do not match the frozen Phase 0 seeds")
    if config.smoke_attempts != 10 or config.main_attempts != 20:
        raise ValueError("attempt counts must be 10 for smoke and 20 for main")
    if config.checkpoint.sha256 != PHASE0_CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash does not match the frozen DF 500K anchor")
    if config.checkpoint.iteration != 500000 or config.checkpoint.state_tensor_count != 392:
        raise ValueError("checkpoint structure does not match the frozen DF 500K anchor")
    if config.interventions != PHASE0_INTERVENTIONS:
        raise ValueError("interventions do not match the frozen D0-D5 matrix")
    if config.d5_gates != (0.25, 0.5, 1.0, 1.5):
        raise ValueError("D5 gates do not match the frozen matrix")
    if config.openness.direction_count != 2048 or config.openness.cutoff_angstrom != 12.0:
        raise ValueError("openness geometry does not match the frozen protocol")
    if config.se3.rotations != 100 or config.se3.translations != 10:
        raise ValueError("SE(3) sample counts do not match the frozen protocol")
    if config.se3.model_float32_tolerance != 1e-4:
        raise ValueError("float32 SE(3) tolerance must be 1e-4")
    if config.se3.analytical_float64_tolerance != 1e-8:
        raise ValueError("float64 analytical tolerance must be 1e-8")
