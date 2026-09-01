#!/usr/bin/env python3
"""Compute create-only openness evidence and freeze smoke/D4 strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import canonical_json, write_new_manifest  # noqa: E402
from dfe.diagnostics.io import sha256_file  # noqa: E402
from dfe.diagnostics.openness import (  # noqa: E402
    compute_openness,
    parse_pdb_heavy_atoms,
    select_smoke_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line]


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_root = manifest_path.parent
    output_path = run_root / "openness.jsonl"
    smoke_path = run_root / "smoke-pockets.json"
    if output_path.exists() or smoke_path.exists():
        raise FileExistsError("openness outputs already exist")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    pockets_path = run_root / manifest["artifacts"]["pockets"]["path"]
    if sha256_file(pockets_path) != manifest["artifacts"]["pockets"]["sha256"]:
        raise ValueError("pockets.jsonl hash does not match the run manifest")

    protocol = manifest["protocol"]["openness"]
    records = []
    for pocket in _read_jsonl(pockets_path):
        pocket_input = next(item for item in pocket["inputs"] if item["role"] == "pocket")
        if sha256_file(Path(pocket_input["source_path"])) != pocket_input["sha256"]:
            raise ValueError(f"pocket input hash mismatch: {pocket['pocket_id']}")
        atoms = parse_pdb_heavy_atoms(Path(pocket_input["source_path"]))
        center = pocket["center"]
        result = compute_openness(
            atoms.coordinates,
            atoms.elements,
            [center["x"], center["y"], center["z"]],
            direction_count=int(protocol["direction_count"]),
            cutoff_angstrom=float(protocol["cutoff_angstrom"]),
            radius_table_version=str(protocol["radius_table_version"]),
        )
        record = {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
            "pocket_id": pocket["pocket_id"],
            **result.to_dict(),
        }
        # Parsing and geometry must agree about fallback accounting.
        record["unknown_element_count"] = atoms.unknown_element_count
        records.append(record)

    smoke, enriched = select_smoke_records(records)
    data = b"".join(canonical_json(record) for record in enriched)
    with output_path.open("xb") as handle:
        handle.write(data)
    write_new_manifest(
        smoke_path,
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
            "selection_ranks": [0, 1, 14, 15, 28, 29],
            "pocket_ids": [record["pocket_id"] for record in smoke],
            "openness_sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    print(f"Created openness evidence for {len(enriched)} pockets and froze 6 smoke pockets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
