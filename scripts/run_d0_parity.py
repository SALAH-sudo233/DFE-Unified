#!/usr/bin/env python3
"""Run one real legacy-vs-D0 attempt and create content-free parity evidence."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import canonical_sha256, write_new_manifest  # noqa: E402
from dfe.diagnostics.io import sha256_file  # noqa: E402
from dfe.diagnostics.ledger import AttemptLedger, replay_ledger  # noqa: E402
from dfe.diagnostics.trace import TraceWriter  # noqa: E402
from sample_diagnostic import (  # noqa: E402
    _build_pocket_data,
    _load_runtime,
    _pocket_by_id,
    _read_jsonl,
    _run_attempt,
    parity_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def _decision_projection(event: Mapping[str, object]) -> dict[str, object]:
    return {
        "step": event["step"],
        "event": event["event"],
        "decision": event["decision"],
    }


def compare_parity_runs(
    baseline: Mapping[str, object], diagnostic: Mapping[str, object]
) -> dict[str, object]:
    checks = {
        "terminal_semantics": baseline["projection"] == diagnostic["projection"],
        "smiles": baseline.get("smiles") == diagnostic.get("smiles"),
        "decision_trace": (
            baseline["decision_hash"] == diagnostic["decision_hash"]
            and baseline["decision_count"] == diagnostic["decision_count"]
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "baseline": {
            "projection": dict(baseline["projection"]),
            "decision_hash": baseline["decision_hash"],
            "decision_count": baseline["decision_count"],
        },
        "diagnostic": {
            "projection": dict(diagnostic["projection"]),
            "decision_hash": diagnostic["decision_hash"],
            "decision_count": diagnostic["decision_count"],
        },
    }


def _run_once(
    temporary_root: Path,
    runtime,
    run_root: Path,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    pocket_record: Mapping[str, object],
    base_data,
    device: str,
    *,
    diagnostics_enabled: bool,
) -> dict[str, object]:
    label = "d0" if diagnostics_enabled else "legacy"
    job_root = temporary_root / label
    job_root.mkdir()
    attempts_path = job_root / "attempts.jsonl"
    events_path = job_root / "events.jsonl"
    from sample_diagnostic import predeclare_attempts

    parity_job = dict(job, attempt_count=1)
    predeclare_attempts(attempts_path, str(manifest["run_id"]), parity_job)
    try:
        with AttemptLedger(attempts_path, resume=True) as ledger, TraceWriter(
            events_path
        ) as trace_writer:
            _run_attempt(
                runtime,
                run_root,
                job_root,
                manifest,
                parity_job,
                0,
                ledger,
                trace_writer,
                pocket_record,
                None,
                base_data,
                None,
                set(),
                device,
                diagnostics_enabled=diagnostics_enabled,
            )
    finally:
        runtime["model"].set_diagnostics(None, None)
    replay = replay_ledger(attempts_path)
    terminal = [
        record
        for record in replay.records
        if record["status"] in {"evaluated", "failed"}
    ]
    if len(terminal) != 1:
        raise ValueError(f"parity run did not create one terminal attempt: {label}")
    decisions = [
        _decision_projection(event)
        for event in _read_jsonl(events_path)
        if event.get("decision") is not None
    ]
    return {
        "projection": parity_projection(terminal[0]),
        "smiles": terminal[0].get("smiles"),
        "decision_hash": canonical_sha256(decisions),
        "decision_count": len(decisions),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    if sha256_file(jobs_path) != manifest["artifacts"]["jobs"]["sha256"]:
        raise ValueError("jobs.jsonl hash does not match the run manifest")
    smoke_ids = json.loads(
        (run_root / "smoke-pockets.json").read_text(encoding="ascii")
    )["pocket_ids"]
    jobs = _read_jsonl(jobs_path)
    job = next(
        item
        for pocket_id in smoke_ids
        for item in jobs
        if item["stage"] == "smoke"
        and item["pocket_id"] == pocket_id
        and item["arm_id"] == "D0"
    )
    runtime = _load_runtime(manifest, args.device)
    pockets = _pocket_by_id(run_root, manifest)
    pocket_record = pockets[str(job["pocket_id"])]
    base_data = _build_pocket_data(runtime, pocket_record)
    with tempfile.TemporaryDirectory(prefix=".d0-parity-", dir=run_root) as temporary:
        temporary_root = Path(temporary)
        baseline = _run_once(
            temporary_root,
            runtime,
            run_root,
            manifest,
            job,
            pocket_record,
            base_data,
            args.device,
            diagnostics_enabled=False,
        )
        diagnostic = _run_once(
            temporary_root,
            runtime,
            run_root,
            manifest,
            job,
            pocket_record,
            base_data,
            args.device,
            diagnostics_enabled=True,
        )
    result = compare_parity_runs(baseline, diagnostic)
    write_new_manifest(
        run_root / "d0-parity.json",
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
            "job_id": job["job_id"],
            "sample_index": 0,
            **result,
        },
    )
    print(f"Created d0-parity.json: {result['status']}")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
