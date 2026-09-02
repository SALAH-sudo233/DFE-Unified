#!/usr/bin/env python3
"""Plan and (when dependencies are available) execute SCI-2A interventions."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import write_new_manifest
from dfe.science.artifact_contract import assert_create_only_output, load_science_manifest
from dfe.science.feature_interventions import INTERVENTIONS


def intervention_names() -> tuple[str, ...]:
    return tuple(item.name for item in INTERVENTIONS)


def denominator_fields() -> tuple[str, ...]:
    return ("attempts", "generated", "reconstructed", "valid", "dockable", "checked")


def build_plan(manifest: dict[str, object], attempts: int) -> dict[str, object]:
    jobs = manifest.get("artifacts", {}).get("jobs", {})
    return {
        "science_experiment_id": "SCI-2A-FEATURE-v1",
        "manifest_hash": manifest.get("manifest_hash"),
        "checkpoint_sha256": manifest.get("inputs", {}).get("checkpoint", {}).get("sha256"),
        "interventions": list(intervention_names()),
        "attempts_per_job": attempts,
        "denominator_fields": list(denominator_fields()),
        "execution_source": "Issue #2 sample_diagnostic.py and ledger; no new scheduler or ledger",
        "jobs_artifact": jobs,
        "status": "planned",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_interventions(args: argparse.Namespace) -> dict[str, object]:
    science_manifest = load_science_manifest(args.manifest)
    science_manifest.require("SCI-2A-FEATURE-v1")
    if args.attempts not in {10, 20}:
        raise ValueError("SCI-2A attempts must be 10 or 20")
    checkpoint = args.checkpoint.resolve()
    expected = science_manifest.checkpoint_sha256
    from dfe.science.artifact_contract import verify_artifact_hash
    verify_artifact_hash(checkpoint, expected)
    return build_plan(dict(science_manifest.payload), args.attempts)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    # Validate the frozen inputs before materializing a new create-only run root.
    report_path = output_root / "sci2a-plan.json"
    try:
        report = run_interventions(args)
        report["dry_run"] = bool(args.dry_run)
    except Exception as exc:
        report = {"science_experiment_id": "SCI-2A-FEATURE-v1", "status": "infrastructure_failure", "error": str(exc), "traceback": traceback.format_exc()}
        output_root.mkdir(parents=True, exist_ok=False)
        write_new_manifest(report_path, report)
        return 1
    output_root.mkdir(parents=True, exist_ok=False)
    write_new_manifest(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
