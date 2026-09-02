#!/usr/bin/env python3
"""Run SCI-1 using the existing Phase 0 SE(3) audit implementation."""

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
from scripts.run_se3_audit import run as run_phase0_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Issue #2 trace root (recorded for provenance)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rotations", type=int, default=100)
    parser.add_argument("--translations", type=int, default=10)
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    manifest = load_science_manifest(args.manifest)
    manifest.require("SCI-1-SE3-v1")
    phase0_args = argparse.Namespace(
        manifest=args.manifest,
        device=args.device,
        rotations=args.rotations,
        translations=args.translations,
        output=None,
    )
    exit_code, report = run_phase0_audit(phase0_args)
    report.update({
        "science_experiment_id": "SCI-1-SE3-v1",
        "manifest_hash": manifest.manifest_sha256,
        "trace_root": str(args.trace) if args.trace else None,
    })
    return exit_code, report


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    assert_create_only_output(output)
    try:
        exit_code, report = run(args)
    except Exception as exc:
        report = {
            "science_experiment_id": "SCI-1-SE3-v1",
            "status": "infrastructure_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_new_manifest(output, report)
        return 1
    write_new_manifest(output, report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
