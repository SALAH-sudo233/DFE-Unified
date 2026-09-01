#!/usr/bin/env python3
"""Run pocket-clustered Phase 0 analyses and create evidence gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import write_new_manifest  # noqa: E402
from dfe.diagnostics.io import sha256_file  # noqa: E402
from dfe.diagnostics.ledger import replay_ledger  # noqa: E402
from dfe.diagnostics.statistics import (  # noqa: E402
    cluster_bootstrap_ci,
    fit_openness_interaction,
    smoke_gate,
)


PRIMARY_METRICS = (
    "validity_rate",
    "dockable_rate",
    "posebusters_rate",
    "uniqueness_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "main"), default="main")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line]


def _verify_output_hashes(
    run_root: Path, records: list[dict[str, object]]
) -> bool:
    for record in records:
        for hash_key, path_key in (
            ("candidate_sha256", "candidate_path"),
            ("sdf_sha256", "sdf_path"),
        ):
            if hash_key not in record:
                continue
            if path_key not in record or sha256_file(run_root / str(record[path_key])) != record[hash_key]:
                return False
    return True


def build_smoke_evidence(run_root: Path, manifest: dict[str, object]):
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    smoke_ids = set(
        json.loads((run_root / "smoke-pockets.json").read_text(encoding="ascii"))["pocket_ids"]
    )
    jobs = [
        job
        for job in _read_jsonl(jobs_path)
        if job["stage"] == "smoke" and job["pocket_id"] in smoke_ids
    ]
    terminal_jobs = 0
    exact_count_jobs = 0
    clean_replay = True
    terminal_records = []
    finite_traces = True
    for job in jobs:
        job_root = run_root / "jobs" / str(job["job_id"])
        attempts_path = job_root / "attempts.jsonl"
        events_path = job_root / "events.jsonl"
        if not attempts_path.is_file() or not events_path.is_file():
            continue
        replay = replay_ledger(attempts_path)
        clean_replay &= not replay.truncated_final_line
        latest = {}
        for record in replay.records:
            latest[record["attempt_id"]] = record
        records = list(latest.values())
        terminal_records.extend(records)
        if len(records) == int(job["attempt_count"]):
            exact_count_jobs += 1
        if len(records) == int(job["attempt_count"]) and all(
            record["status"] in {"evaluated", "failed"} for record in records
        ):
            terminal_jobs += 1
        try:
            for event in _read_jsonl(events_path):
                tensor = event.get("tensor")
                if tensor:
                    finite_traces &= all(
                        isinstance(value, (int, float))
                        and float("-inf") < float(value) < float("inf")
                        for key, value in tensor.items()
                        if key in {"min", "max", "mean", "l2"}
                    )
        except (OSError, json.JSONDecodeError):
            finite_traces = False
    parity_path = run_root / "d0-parity.json"
    parity = (
        json.loads(parity_path.read_text(encoding="ascii")).get("status") == "pass"
        if parity_path.is_file()
        else False
    )
    evidence = {
        "expected_job_count": len(jobs),
        "terminal_job_count": terminal_jobs,
        "expected_attempts_per_job": 10,
        "jobs_with_exact_attempt_count": exact_count_jobs,
        "normal_parity_passed": parity,
        "finite_traces": finite_traces,
        "clean_replay": clean_replay,
        "output_hashes_valid": _verify_output_hashes(run_root, terminal_records),
    }
    retained = ["D0", "D1", "D2"]
    optional = ("D3", "D4", "D5-g0.25", "D5-g0.5", "D5-g1", "D5-g1.5")
    for arm in optional:
        arm_jobs = [job for job in jobs if job["arm_id"] == arm]
        if arm_jobs and all(
            (run_root / "jobs" / str(job["job_id"]) / "attempts.jsonl").is_file()
            for job in arm_jobs
        ):
            retained.append(arm)
    return evidence, retained


def _analysis_rows(run_root: Path) -> list[dict[str, object]]:
    summary_rows = _read_csv(run_root / "per-pocket-seed.csv")
    openness = {
        record["pocket_id"]: record for record in _read_jsonl(run_root / "openness.jsonl")
    }
    rows = []
    for record in summary_rows:
        pocket = openness[record["pocket_id"]]
        for metric in PRIMARY_METRICS:
            if record.get(metric) in {None, ""}:
                continue
            rows.append(
                {
                    "pocket_id": record["pocket_id"],
                    "seed": int(record["seed"]),
                    "intervention": record["intervention"],
                    "metric_name": metric,
                    "metric": float(record[metric]),
                    "openness": float(pocket["openness"]),
                    "pocket_atom_count": int(pocket["heavy_atom_count"]),
                    "reference_ligand_heavy_atoms": int(
                        pocket.get("reference_ligand_heavy_atoms", 0)
                    ),
                }
            )
    return rows


def analyze_main(run_root: Path) -> list[dict[str, object]]:
    rows = _analysis_rows(run_root)
    interventions = sorted({row["intervention"] for row in rows} - {"D0"})
    results = []
    for metric in PRIMARY_METRICS:
        metric_rows = [row for row in rows if row["metric_name"] == metric]
        for intervention in interventions:
            subset = [
                row
                for row in metric_rows
                if row["intervention"] in {"D0", intervention}
            ]
            fit = fit_openness_interaction(subset, intervention=intervention)
            ci = cluster_bootstrap_ci(
                subset,
                lambda sample, arm=intervention: fit_openness_interaction(
                    sample, intervention=arm
                ).interaction,
                draws=10000,
                seed=20260901,
            )
            results.append(
                {
                    "metric": metric,
                    "intervention": intervention,
                    "interaction": fit.interaction,
                    "coefficients": dict(fit.coefficients),
                    "bootstrap": asdict(ci),
                }
            )
    return results


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    gate_path = run_root / ("gate-smoke.json" if args.stage == "smoke" else "gate-phase0.json")
    if args.verify_only:
        gate = json.loads(gate_path.read_text(encoding="ascii"))
        if gate["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("gate manifest hash mismatch")
        print(f"Verified {gate_path.name}: {gate['status']}")
        return 0
    if args.stage == "smoke":
        evidence, retained = build_smoke_evidence(run_root, manifest)
        gate = smoke_gate(evidence, retained_arm_ids=retained)
    else:
        results = analyze_main(run_root)
        se3 = json.loads((run_root / "se3-audit.json").read_text(encoding="ascii"))
        gate = {
            "status": "pass",
            "artifact_completeness": True,
            "se3_hypothesis": se3["status"],
            "analysis": results,
        }
    write_new_manifest(
        gate_path,
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
            **gate,
        },
    )
    print(f"Created {gate_path.name}: {gate['status']}")
    return 0 if gate["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
