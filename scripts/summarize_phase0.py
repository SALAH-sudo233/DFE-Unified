#!/usr/bin/env python3
"""Summarize Phase 0 ledgers with requested-attempt denominators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def summarize(manifest_path: Path) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    if sha256_file(jobs_path) != manifest["artifacts"]["jobs"]["sha256"]:
        raise ValueError("jobs.jsonl hash mismatch")
    jobs = {
        record["job_id"]: record
        for record in (
            json.loads(line) for line in jobs_path.read_text(encoding="ascii").splitlines()
        )
    }
    attempts = []
    for job_root in sorted((run_root / "jobs").iterdir()):
        if not job_root.is_dir() or job_root.name not in jobs:
            continue
        replay = replay_ledger(job_root / "attempts.jsonl")
        if replay.truncated_final_line:
            raise ValueError(f"truncated attempt ledger: {job_root.name}")
        terminal = terminal_attempt_records(replay.records)
        expected = int(jobs[job_root.name]["attempt_count"])
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
        "attempts": attempts,
        "pocket_seed_rows": pocket_seed_rows,
        "pocket_rows": pocket_rows,
        "summary": total,
    }


def verify_outputs(run_root: Path, manifest_hash: str) -> None:
    summary_path = run_root / "phase0-summary.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if summary["manifest_hash"] != manifest_hash:
        raise ValueError("summary manifest hash mismatch")
    if summary["metric_definition_version"] != METRIC_DEFINITION_VERSION:
        raise ValueError("summary metric definition mismatch")
    for name in (
        "per-attempt.parquet",
        "per-pocket-seed.csv",
        "per-pocket.csv",
        "failure-taxonomy.csv",
    ):
        if not (run_root / name).is_file():
            raise ValueError(f"missing summary artifact: {name}")


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if args.verify_only:
        verify_outputs(manifest_path.parent, manifest["manifest_hash"])
        print("Phase 0 summary artifacts verified.")
        return 0
    result = summarize(manifest_path)
    run_root = manifest_path.parent
    _write_parquet(run_root / "per-attempt.parquet", result["attempts"])
    _write_csv(run_root / "per-pocket-seed.csv", result["pocket_seed_rows"])
    _write_csv(run_root / "per-pocket.csv", result["pocket_rows"])
    _write_csv(
        run_root / "failure-taxonomy.csv",
        [
            {"failure_code": key, "count": value}
            for key, value in result["summary"]["failure_taxonomy"].items()
        ],
    )
    write_new_manifest(
        run_root / "phase0-summary.json",
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
            **result["summary"],
        },
    )
    print(f"Summarized {len(result['attempts'])} terminal attempts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
