#!/usr/bin/env python3
"""Generate deterministic provenance manifests for the curated repository."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOWNLOAD_RECORDS = ROOT / "evidence" / "download-records.json"
RESULT_ROOT = ROOT / "results" / "df-500k-21-pocket"
UPSTREAM_COMMIT = "836a0c4ce487297ad24bc54ac2ebd163de13242c"
CHECKPOINT_SHA256 = "34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_download_records() -> list[dict[str, Any]]:
    payload = json.loads(DOWNLOAD_RECORDS.read_text(encoding="utf-8"))
    return payload["files"]


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / record["path"]
    if not path.is_file():
        raise FileNotFoundError(record["path"])
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    if actual_size != record["size"]:
        raise ValueError(f"size mismatch for {record['path']}: {actual_size} != {record['size']}")
    if actual_hash != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {record['path']}")
    return {
        "path": record["path"],
        "remote_path": record["remote_path"],
        "remote_mtime_epoch": record["mtime"],
        "sha256": actual_hash,
        "size": actual_size,
    }


def build_code_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    code_records = [
        validate_record(record)
        for record in records
        if not record["path"].startswith(("artifacts/", "results/"))
    ]
    return {
        "description": "Allowlisted files retrieved from the remote tree used by the DF 500K run.",
        "source_repository": "https://github.com/PengXingang/Pocket2Mol",
        "upstream_commit": UPSTREAM_COMMIT,
        "files": sorted(code_records, key=lambda item: item["path"]),
    }


def build_artifact_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint_record = next(
        validate_record(record)
        for record in records
        if record["path"] == "artifacts/checkpoints/df-500k.pt"
    )
    if checkpoint_record["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("DF 500K checkpoint does not match the independently observed hash")
    log_record = next(
        validate_record(record)
        for record in records
        if record["path"] == "artifacts/logs/train_df_resume_380k.log"
    )
    return {
        "checkpoint": {
            **checkpoint_record,
            "iteration": 500000,
            "status": "observed-trained-artifact",
        },
        "training_log": log_record,
    }


def build_result_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    raw_paths = sorted((RESULT_ROOT / "per-pocket").glob("*/docking_results.json"))
    raw_results = {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in raw_paths}
    requested = json.loads((ROOT / "pocket_centers_30.json").read_text(encoding="utf-8"))
    completed = sorted(summary)
    missing = sorted(set(requested) - set(completed))
    evaluated_records = sum(len(items) for items in raw_results.values())
    posebusters_passed = sum(
        (item.get("posebuster") or {}).get("passed") is True
        for items in raw_results.values()
        for item in items
    )
    docked_below_minus_seven = sum(
        item.get("docking_score") is not None and item["docking_score"] < -7.0
        for items in raw_results.values()
        for item in items
    )
    result_records = [
        validate_record(record)
        for record in records
        if record["path"].startswith("results/")
    ]
    return {
        "status": "partial",
        "requested_pockets": len(requested),
        "completed_pockets": len(completed),
        "completed_pocket_ids": completed,
        "missing_pocket_ids": missing,
        "evaluated_records": evaluated_records,
        "macro_metrics": {
            "docking_score_mean": sum(item["docking_score_mean"] for item in summary.values()) / len(summary),
            "qed_mean": sum(item["qed_mean"] for item in summary.values()) / len(summary),
            "molecular_weight_mean": sum(item["mw_mean"] for item in summary.values()) / len(summary),
        },
        "posebusters": {
            "passed": posebusters_passed,
            "pass_rate_over_evaluated_records": posebusters_passed / evaluated_records,
        },
        "docking_score_below_minus_7": {
            "records": docked_below_minus_seven,
            "rate_over_evaluated_records": docked_below_minus_seven / evaluated_records,
        },
        "validity_denominator_warning": (
            "The retained evaluator loaded already reconstructed SDF molecules. Reconstruction "
            "failures and molecules that could not be written/read as SDF are not represented in "
            "this denominator, so the reported success_rate is not end-to-end generation validity."
        ),
        "files": sorted(result_records, key=lambda item: item["path"]),
    }


def main() -> None:
    records = load_download_records()
    write_json(ROOT / "evidence" / "code-manifest.json", build_code_manifest(records))
    write_json(ROOT / "artifacts" / "MANIFEST.json", build_artifact_manifest(records))
    write_json(RESULT_ROOT / "provenance.json", build_result_provenance(records))
    print("Generated code, artifact, and result provenance manifests.")


if __name__ == "__main__":
    main()
