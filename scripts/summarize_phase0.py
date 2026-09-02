#!/usr/bin/env python3
"""Summarize Phase 0 ledgers with requested-attempt denominators."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import write_new_manifest  # noqa: E402
from dfe.diagnostics.io import sha256_file  # noqa: E402
from dfe.diagnostics.ledger import replay_ledger  # noqa: E402
from dfe.diagnostics.metrics import (  # noqa: E402
    METRIC_DEFINITION_VERSION,
    aggregation_key,
    summarize_attempts,
    terminal_attempt_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "main"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary(keys: dict[str, object], summary: dict[str, object]):
    row = {**keys, "attempt_count": summary["attempt_count"]}
    for metric in ("validity", "dockable", "posebusters", "uniqueness"):
        for name, value in summary[metric].items():
            row[f"{metric}_{name}"] = value
    return row


def _write_parquet(path: Path, records: list[dict[str, object]]) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("a NumPy-compatible pandas/parquet engine is required") from exc
    try:
        pd.DataFrame(records).to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError("failed to write real Parquet output") from exc


def summary_root(run_root: Path, stage: str) -> Path:
    if stage not in {"smoke", "main"}:
        raise ValueError(f"unknown summary stage: {stage}")
    return Path(run_root) / "summaries" / stage


def selected_jobs(
    run_root: Path, jobs: list[dict[str, object]], stage: str
) -> list[dict[str, object]]:
    selected = [job for job in jobs if job["stage"] == stage]
    if stage == "smoke":
        smoke = json.loads(
            (run_root / "smoke-pockets.json").read_text(encoding="ascii")
        )
        pocket_ids = set(smoke["pocket_ids"])
        return [job for job in selected if job["pocket_id"] in pocket_ids]
    gate = json.loads((run_root / "gate-smoke.json").read_text(encoding="ascii"))
    if gate.get("status") != "pass":
        raise ValueError("main summary requires a passing smoke gate")
    retained = set(gate["retained_arm_ids"])
    return [job for job in selected if job["arm_id"] in retained]


def summarize(manifest_path: Path, stage: str) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    if sha256_file(jobs_path) != manifest["artifacts"]["jobs"]["sha256"]:
        raise ValueError("jobs.jsonl hash mismatch")
    declared = [
        json.loads(line)
        for line in jobs_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    jobs = {
        record["job_id"]: record
        for record in selected_jobs(run_root, declared, stage)
    }
    if not jobs:
        raise ValueError(f"no jobs declared for {stage} summary")
    attempts = []
    for job_id, job in sorted(jobs.items()):
        job_root = run_root / "jobs" / str(job_id)
        if not job_root.is_dir():
            raise ValueError(f"missing job directory: {job_id}")
        replay = replay_ledger(job_root / "attempts.jsonl")
        if replay.truncated_final_line:
            raise ValueError(f"truncated attempt ledger: {job_root.name}")
        terminal = terminal_attempt_records(replay.records)
        expected = int(job["attempt_count"])
        if len(terminal) != expected:
            raise ValueError(f"attempt count mismatch for {job_root.name}")
        attempts.extend(terminal)
    if not attempts:
        raise ValueError("no terminal attempts found")

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in attempts:
        grouped[aggregation_key(record)].append(record)
    pocket_seed_rows = [
        _flatten_summary(
            {"pocket_id": key[0], "seed": key[1], "intervention": key[2]},
            summarize_attempts(values),
        )
        for key, values in sorted(grouped.items())
    ]
    pocket_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in attempts:
        pocket_groups[(record["pocket_id"], record["arm_id"])].append(record)
    pocket_rows = [
        _flatten_summary(
            {"pocket_id": key[0], "intervention": key[1]},
            summarize_attempts(values),
        )
        for key, values in sorted(pocket_groups.items())
    ]
    total = summarize_attempts(attempts)
    return {
        "manifest": manifest,
        "stage": stage,
        "attempts": attempts,
        "pocket_seed_rows": pocket_seed_rows,
        "pocket_rows": pocket_rows,
        "summary": total,
    }


def verify_outputs(run_root: Path, manifest_hash: str, stage: str) -> None:
    output_root = summary_root(run_root, stage)
    summary_path = output_root / "phase0-summary.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if summary["manifest_hash"] != manifest_hash:
        raise ValueError("summary manifest hash mismatch")
    if summary["metric_definition_version"] != METRIC_DEFINITION_VERSION:
        raise ValueError("summary metric definition mismatch")
    if summary["stage"] != stage:
        raise ValueError("summary stage mismatch")
    for name in (
        "per-attempt.parquet",
        "per-pocket-seed.csv",
        "per-pocket.csv",
        "failure-taxonomy.csv",
    ):
        path = output_root / name
        if not path.is_file():
            raise ValueError(f"missing summary artifact: {name}")
        artifact = summary.get("artifacts", {}).get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"missing summary artifact hash: {name}")
        if artifact.get("size") != path.stat().st_size:
            raise ValueError(f"summary artifact size mismatch: {name}")
        if artifact.get("sha256") != sha256_file(path):
            raise ValueError(f"summary artifact hash mismatch: {name}")


def write_outputs(output_root: Path, result: dict[str, object]) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise FileExistsError(f"summary output already exists: {output_root}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    try:
        _write_parquet(staging / "per-attempt.parquet", result["attempts"])
        _write_csv(staging / "per-pocket-seed.csv", result["pocket_seed_rows"])
        _write_csv(staging / "per-pocket.csv", result["pocket_rows"])
        _write_csv(
            staging / "failure-taxonomy.csv",
            [
                {"failure_code": key, "count": value}
                for key, value in result["summary"]["failure_taxonomy"].items()
            ],
        )
        artifact_names = (
            "per-attempt.parquet",
            "per-pocket-seed.csv",
            "per-pocket.csv",
            "failure-taxonomy.csv",
        )
        write_new_manifest(
            staging / "phase0-summary.json",
            {
                "schema_version": result["manifest"]["schema_version"],
                "manifest_hash": result["manifest"]["manifest_hash"],
                "stage": result["stage"],
                "artifacts": {
                    name: {
                        "size": (staging / name).stat().st_size,
                        "sha256": sha256_file(staging / name),
                    }
                    for name in artifact_names
                },
                **result["summary"],
            },
        )
        os.replace(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if args.verify_only:
        verify_outputs(manifest_path.parent, manifest["manifest_hash"], args.stage)
        print(f"Phase 0 {args.stage} summary artifacts verified.")
        return 0
    result = summarize(manifest_path, args.stage)
    write_outputs(summary_root(manifest_path.parent, args.stage), result)
    print(f"Summarized {len(result['attempts'])} terminal {args.stage} attempts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
