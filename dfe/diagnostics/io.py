"""Hash-bound input resolution and manifest construction for Phase 0."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    PHASE0_CHECKPOINT_SHA256,
    canonical_json,
    canonical_sha256,
    load_phase0_config,
    write_new_manifest,
)


class InputManifestError(ValueError):
    """Raised when frozen inputs cannot support a reproducible run manifest."""


@dataclass(frozen=True)
class InputFileRecord:
    role: str
    logical_path: str
    source_path: Path
    size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source_path"] = str(self.source_path)
        return value


@dataclass(frozen=True)
class GitState:
    commit: str
    branch: str | None
    is_dirty: bool
    porcelain: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(role: str, logical_path: str, source_path: Path) -> InputFileRecord:
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise InputManifestError(f"missing {role} input: {source_path}")
    return InputFileRecord(
        role=role,
        logical_path=Path(logical_path).as_posix(),
        source_path=source_path,
        size=source_path.stat().st_size,
        sha256=sha256_file(source_path),
    )


def resolve_pdbbind_inputs(root: Path, pocket_id: str) -> tuple[InputFileRecord, ...]:
    root = Path(root).resolve()
    directory = root / pocket_id
    specs = (
        ("protein", f"{pocket_id}_protein.pdb"),
        ("pocket", f"{pocket_id}_pocket.pdb"),
        ("ligand", f"{pocket_id}_ligand.sdf"),
    )
    return tuple(
        _file_record(role, f"{pocket_id}/{filename}", directory / filename)
        for role, filename in specs
    )


def _run_git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def git_state(repository_root: Path) -> GitState:
    repository_root = Path(repository_root).resolve()
    try:
        commit = _run_git(repository_root, "rev-parse", "HEAD")
        branch_value = _run_git(repository_root, "branch", "--show-current")
        porcelain_value = _run_git(
            repository_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputManifestError(f"cannot inspect Git state: {repository_root}") from exc
    porcelain = tuple(line for line in porcelain_value.splitlines() if line)
    return GitState(commit, branch_value or None, bool(porcelain), porcelain)


def _load_centers(path: Path, expected_count: int) -> list[tuple[str, tuple[float, ...]]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputManifestError(f"cannot read pocket centers: {path}") from exc
    if not isinstance(value, dict):
        raise InputManifestError("pocket centers must be a JSON mapping")
    if len(value) != expected_count:
        raise InputManifestError(
            f"expected {expected_count} pocket centers, found {len(value)}"
        )
    records: list[tuple[str, tuple[float, ...]]] = []
    for pocket_id, center in value.items():
        if not isinstance(pocket_id, str) or not pocket_id.strip():
            raise InputManifestError("pocket IDs must be non-empty strings")
        if not isinstance(center, list) or len(center) != 3:
            raise InputManifestError(f"pocket center must contain xyz: {pocket_id}")
        try:
            numeric = tuple(float(component) for component in center)
        except (TypeError, ValueError) as exc:
            raise InputManifestError(f"pocket center must be numeric: {pocket_id}") from exc
        records.append((pocket_id, numeric))
    return sorted(records)


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(record) for record in records)


def _artifact_record(path: Path, data: bytes) -> dict[str, object]:
    return {
        "path": path.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "yaml", "torch", "rdkit"):
        try:
            module = __import__(module_name)
        except ImportError:
            versions[module_name] = "unavailable"
        else:
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _arm_specs(d5_gates: tuple[float, ...]) -> tuple[dict[str, object], ...]:
    fixed = tuple(
        {"arm_id": intervention, "intervention": intervention, "gate": 1.0}
        for intervention in ("D0", "D1", "D2", "D3", "D4")
    )
    scaled = tuple(
        {
            "arm_id": f"D5-g{gate:g}",
            "intervention": "D5",
            "gate": gate,
        }
        for gate in d5_gates
    )
    return fixed + scaled


def _job_records(
    pocket_ids: Iterable[str],
    seeds: tuple[int, ...],
    smoke_attempts: int,
    main_attempts: int,
    arms: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    frozen_pocket_ids = tuple(sorted(pocket_ids))
    jobs: list[dict[str, object]] = []
    for stage, stage_seeds, attempt_count in (
        ("smoke", seeds[:1], smoke_attempts),
        ("main", seeds, main_attempts),
    ):
        for pocket_id in frozen_pocket_ids:
            for seed in stage_seeds:
                for arm in arms:
                    arm_id = str(arm["arm_id"])
                    jobs.append(
                        {
                            "schema_version": "phase0.v1",
                            "job_id": f"{stage}:{pocket_id}:{seed}:{arm_id}",
                            "stage": stage,
                            "pocket_id": pocket_id,
                            "seed": seed,
                            "attempt_count": attempt_count,
                            **arm,
                        }
                    )
    return jobs


def build_phase0_manifest(
    config_path: Path,
    pdbbind_root: Path,
    centers_path: Path,
    output_root: Path,
    *,
    repository_root: Path | None = None,
    require_clean: bool = False,
    expected_pocket_count: int = 30,
) -> dict[str, object]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    repository_root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    config_path = Path(config_path).resolve()
    centers_path = Path(centers_path).resolve()
    config = load_phase0_config(config_path)
    state = git_state(repository_root)
    if require_clean and state.is_dirty:
        raise InputManifestError("--require-clean requires a clean Git tree")

    centers = _load_centers(centers_path, expected_pocket_count)
    pocket_records = []
    for pocket_id, center in centers:
        inputs = resolve_pdbbind_inputs(pdbbind_root, pocket_id)
        pocket_records.append(
            {
                "schema_version": "phase0.v1",
                "pocket_id": pocket_id,
                "center": {"x": center[0], "y": center[1], "z": center[2]},
                "inputs": [record.to_dict() for record in inputs],
            }
        )

    checkpoint = _file_record(
        "checkpoint", "artifacts/checkpoints/df-500k.pt", config.checkpoint.path
    )
    if checkpoint.sha256 != PHASE0_CHECKPOINT_SHA256:
        raise InputManifestError("checkpoint hash does not match the frozen DF 500K anchor")
    config_record = _file_record(
        "config", "configs/diagnostics/phase0_df500k.yaml", config_path
    )
    sampling_policy_record = _file_record(
        "sampling_policy", "configs/sample_df_500k.yml", config.sampling_policy
    )
    centers_record = _file_record("centers", "pocket_centers_30.json", centers_path)

    arms = _arm_specs(config.d5_gates)
    jobs = _job_records(
        (record["pocket_id"] for record in pocket_records),
        config.seeds,
        config.smoke_attempts,
        config.main_attempts,
        arms,
    )
    pockets_bytes = _jsonl_bytes(pocket_records)
    jobs_bytes = _jsonl_bytes(jobs)
    pockets_artifact = _artifact_record(output_root / "pockets.jsonl", pockets_bytes)
    jobs_artifact = _artifact_record(output_root / "jobs.jsonl", jobs_bytes)
    smoke_count = sum(job["stage"] == "smoke" for job in jobs)
    main_count = sum(job["stage"] == "main" for job in jobs)

    manifest: dict[str, object] = {
        "schema_version": config.schema_version,
        "run_id": config.experiment_ids["manifest"],
        "manifest_hash": "",
        "pocket_count": len(pocket_records),
        "smoke_candidate_job_count": smoke_count,
        "main_candidate_job_count": main_count,
        "smoke_selection_status": "pending_openness",
        "git": state.to_dict(),
        "environment": _environment_versions(),
        "inputs": {
            "config": config_record.to_dict(),
            "sampling_policy": sampling_policy_record.to_dict(),
            "centers": centers_record.to_dict(),
            "checkpoint": checkpoint.to_dict(),
        },
        "protocol": {
            "seeds": config.seeds,
            "smoke_attempts": config.smoke_attempts,
            "main_attempts": config.main_attempts,
            "arms": arms,
            "openness": asdict(config.openness),
            "se3": asdict(config.se3),
        },
        "artifacts": {"pockets": pockets_artifact, "jobs": jobs_artifact},
    }
    manifest["manifest_hash"] = canonical_sha256(
        {**manifest, "manifest_hash": ""}
    )

    # No run path is created until every input and output byte has been validated.
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        with (output_root / "pockets.jsonl").open("xb") as handle:
            handle.write(pockets_bytes)
        with (output_root / "jobs.jsonl").open("xb") as handle:
            handle.write(jobs_bytes)
        write_new_manifest(output_root / "run-manifest.json", manifest)
    except Exception:
        # Preserve create-only semantics: do not rewrite or silently reuse partial evidence.
        raise
    return manifest
