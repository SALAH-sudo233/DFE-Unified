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
    phase0_gate,
    smoke_gate,
)
from scripts.summarize_phase0 import verify_outputs  # noqa: E402


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
        successful = (
            record.get("status") == "evaluated"
            and record.get("evaluation_status") != "failed"
            and bool(record.get("smiles"))
        )
        for hash_key, path_key in (
            ("candidate_sha256", "candidate_path"),
            ("sdf_sha256", "sdf_path"),
        ):
            if not successful and hash_key not in record and path_key not in record:
                continue
            if hash_key not in record or path_key not in record:
                return False
            path = run_root / str(record[path_key])
            if not path.is_file() or sha256_file(path) != record[hash_key]:
                return False
    return True


def _job_evidence(run_root: Path, job: dict[str, object]):
    job_root = run_root / "jobs" / str(job["job_id"])
    attempts_path = job_root / "attempts.jsonl"
    events_path = job_root / "events.jsonl"
    if not attempts_path.is_file() or not events_path.is_file():
        return None
    try:
        replay = replay_ledger(attempts_path)
        events = _read_jsonl(events_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    latest = {record["attempt_id"]: record for record in replay.records}
    records = list(latest.values())
    exact = len(records) == int(job["attempt_count"])
    terminal = exact and all(
        record["status"] in {"evaluated", "failed"} for record in records
    )
    finite = True
    for event in events:
        tensor = event.get("tensor")
        if tensor:
            finite &= all(
                isinstance(value, (int, float))
                and float("-inf") < float(value) < float("inf")
                for key, value in tensor.items()
                if key in {"min", "max", "mean", "l2"}
            )
    return {
        "records": records,
        "exact": exact,
        "terminal": terminal,
        "clean": not replay.truncated_final_line,
        "finite": finite,
    }


def complete_arm_ids(run_root: Path, jobs: list[dict[str, object]]) -> set[str]:
    by_arm: dict[str, list[dict[str, object]]] = {}
    for job in jobs:
        by_arm.setdefault(str(job["arm_id"]), []).append(job)
    complete = set()
    for arm_id, arm_jobs in by_arm.items():
        evidence = [_job_evidence(run_root, job) for job in arm_jobs]
        if evidence and all(
            item is not None
            and item["exact"]
            and item["terminal"]
            and item["clean"]
            and item["finite"]
            for item in evidence
        ):
            complete.add(arm_id)
    return complete


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
        item = _job_evidence(run_root, job)
        if item is None:
            continue
        records = item["records"]
        terminal_records.extend(records)
        clean_replay &= item["clean"]
        finite_traces &= item["finite"]
        if item["exact"]:
            exact_count_jobs += 1
        if item["terminal"]:
            terminal_jobs += 1
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
    complete_arms = complete_arm_ids(run_root, jobs)
    for arm in optional:
        if arm in complete_arms:
            retained.append(arm)
    return evidence, retained


def build_main_evidence(run_root: Path, manifest: dict[str, object]):
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    smoke_gate_record = json.loads(
        (run_root / "gate-smoke.json").read_text(encoding="ascii")
    )
    retained = set(smoke_gate_record["retained_arm_ids"])
    jobs = [
        job
        for job in _read_jsonl(jobs_path)
        if job["stage"] == "main" and job["arm_id"] in retained
    ]
    evidence_items = [_job_evidence(run_root, job) for job in jobs]
    records = [
        record
        for item in evidence_items
        if item is not None
        for record in item["records"]
    ]
    summary_root = run_root / "summaries" / "main"
    summary_path = summary_root / "phase0-summary.json"
    summary_valid = False
    if summary_path.is_file():
        try:
            verify_outputs(run_root, str(manifest["manifest_hash"]), "main")
            summary_valid = True
        except (OSError, ValueError, json.JSONDecodeError):
            summary_valid = False
    return {
        "expected_job_count": len(jobs),
        "terminal_job_count": sum(
            item is not None and item["terminal"] for item in evidence_items
        ),
        "expected_attempts_per_job": 20,
        "jobs_with_exact_attempt_count": sum(
            item is not None and item["exact"] for item in evidence_items
        ),
        "finite_traces": all(
            item is not None and item["finite"] for item in evidence_items
        ),
        "clean_replay": all(
            item is not None and item["clean"] for item in evidence_items
        ),
        "output_hashes_valid": _verify_output_hashes(run_root, records),
        "summary_artifacts_valid": summary_valid,
    }


def _analysis_rows(run_root: Path) -> list[dict[str, object]]:
    summary_rows = _read_csv(run_root / "summaries" / "main" / "per-pocket-seed.csv")
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
        evidence = build_main_evidence(run_root, manifest)
        results = analyze_main(run_root) if evidence["summary_artifacts_valid"] else []
        se3 = json.loads((run_root / "se3-audit.json").read_text(encoding="ascii"))
        gate = phase0_gate(
            evidence,
            se3_hypothesis=str(se3["status"]),
            analysis=results,
        )
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
