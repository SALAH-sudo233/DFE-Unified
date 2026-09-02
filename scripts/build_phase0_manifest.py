#!/usr/bin/env python3
"""Create the hash-bound input and candidate-job manifest for Phase 0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.io import build_phase0_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pdbbind-root", type=Path, required=True)
    parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_phase0_manifest(
        args.config,
        args.pdbbind_root,
        args.centers,
        args.output_root,
        repository_root=ROOT,
        require_clean=args.require_clean,
    )
    print(
        f"Created {args.output_root} with {manifest['pocket_count']} pockets, "
        f"{manifest['smoke_candidate_job_count']} smoke candidates and "
        f"{manifest['main_candidate_job_count']} main candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
