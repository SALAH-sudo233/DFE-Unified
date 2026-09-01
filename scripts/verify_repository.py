#!/usr/bin/env python3
"""Verify the integrity, scope, and publication safety of this evidence snapshot."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys
import types
from collections.abc import Iterable
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = (
    "34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0"
)
FORBIDDEN_NAME_FRAGMENTS = (
    "ssh_",
    "paramiko",
    "heartbeat",
    "credentials",
    "private_key",
)
SECRET_PATTERNS = (
    ("private key material", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "secret assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|api_key|access_token|secret_key)\b\s*[:=]\s*"
            r"[\"'][^\"'\r\n]{4,}[\"']"
        ),
    ),
    ("Paramiko client", re.compile(r"paramiko\s*\.\s*SSHClient\s*\(")),
    ("credential decoding", re.compile(r"(?i)\b(?:password|passwd|pw)\b.*bytes\.fromhex\s*\(")),
    ("known tunnel endpoint", re.compile(r"\b21\.tcp\.cpolar\.top\b")),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def tracked_files() -> list[pathlib.Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def relative_posix(root: pathlib.Path, path: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def scan_forbidden_names(
    root: pathlib.Path, paths: Iterable[pathlib.Path]
) -> list[str]:
    violations = []
    for path in paths:
        name = relative_posix(root, path).lower()
        if any(fragment in name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            violations.append(f"forbidden filename: {name}")
    return violations


def scan_forbidden_content(
    root: pathlib.Path, paths: Iterable[pathlib.Path]
) -> list[str]:
    violations = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for description, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                violations.append(
                    f"{description}: {relative_posix(root, path)}"
                )
    return violations


def verify_manifest(
    path: pathlib.Path,
    records: Iterable[dict[str, Any]],
    *,
    repository_bytes: bool = False,
) -> None:
    for record in records:
        candidate = ROOT / record["path"]
        if not candidate.is_file():
            raise ValueError(f"{path}: missing {record['path']}")
        size_key = "repository_size" if repository_bytes and "repository_size" in record else "size"
        hash_key = (
            "repository_sha256"
            if repository_bytes and "repository_sha256" in record
            else "sha256"
        )
        if candidate.stat().st_size != record[size_key]:
            raise ValueError(f"{path}: size mismatch for {record['path']}")
        if sha256(candidate) != record[hash_key]:
            raise ValueError(f"{path}: SHA-256 mismatch for {record['path']}")


def verify_manifests() -> None:
    code_path = ROOT / "evidence" / "code-manifest.json"
    artifact_path = ROOT / "artifacts" / "MANIFEST.json"
    result_path = ROOT / "results" / "df-500k-21-pocket" / "provenance.json"
    code = json.loads(code_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    results = json.loads(result_path.read_text(encoding="utf-8"))
    verify_manifest(code_path, code["files"], repository_bytes=True)
    verify_manifest(
        artifact_path, [artifact["checkpoint"], artifact["training_log"]]
    )
    verify_manifest(result_path, results["files"])
    if artifact["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("unexpected checkpoint SHA-256 in artifact manifest")


def verify_json() -> int:
    count = 0
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts:
            continue
        json.loads(path.read_text(encoding="utf-8"))
        count += 1
    return count


def verify_results() -> None:
    result_root = ROOT / "results" / "df-500k-21-pocket"
    provenance = json.loads((result_root / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    result_paths = sorted((result_root / "per-pocket").glob("*/docking_results.json"))
    record_count = sum(
        len(json.loads(path.read_text(encoding="utf-8"))) for path in result_paths
    )
    observed = (
        provenance["requested_pockets"],
        provenance["completed_pockets"],
        provenance["evaluated_records"],
        len(summary),
        len(result_paths),
        record_count,
    )
    if observed != (30, 21, 2331, 21, 21, 2331):
        raise ValueError(f"unexpected partial-run scope: {observed}")


def install_easydict_pickle_shim() -> None:
    module = types.ModuleType("easydict")

    class EasyDict(dict):
        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        __setattr__ = dict.__setitem__

    EasyDict.__module__ = "easydict"
    EasyDict.__qualname__ = "EasyDict"
    module.EasyDict = EasyDict
    sys.modules.setdefault("easydict", module)


def verify_checkpoint() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect the checkpoint") from exc

    install_easydict_pickle_shim()
    path = ROOT / "artifacts" / "checkpoints" / "df-500k.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("iteration") != 500000:
        raise ValueError(f"unexpected checkpoint iteration: {checkpoint.get('iteration')}")
    if set(checkpoint) != {"config", "iteration", "model", "optimizer", "scheduler"}:
        raise ValueError(f"unexpected checkpoint keys: {sorted(checkpoint)}")
    if len(checkpoint["model"]) != 392:
        raise ValueError(f"unexpected model state tensor count: {len(checkpoint['model'])}")


def verify_lfs(paths: Iterable[pathlib.Path]) -> None:
    relative_paths = [relative_posix(ROOT, path) for path in paths]
    for required in (
        "artifacts/checkpoints/df-500k.pt",
        "artifacts/logs/train_df_resume_380k.log",
    ):
        attribute = run_git("check-attr", "filter", "--", required).strip()
        if not attribute.endswith(": lfs"):
            raise ValueError(f"Git LFS filter is not configured for {required}")
    for path, relative in zip(paths, relative_paths):
        if path.stat().st_size <= 100 * 1024 * 1024:
            continue
        attribute = run_git("check-attr", "filter", "--", relative).strip()
        if not attribute.endswith(": lfs"):
            raise ValueError(f"file over 100 MiB is not tracked by LFS: {relative}")


def verify_sdf() -> str:
    try:
        from rdkit import Chem
    except ImportError:
        return "RDKit unavailable; optional SDF parsing skipped"

    sdf_paths = sorted(
        (ROOT / "results" / "df-500k-21-pocket" / "per-pocket").glob("*/merged_all.sdf")
    )
    molecule_count = 0
    invalid_count = 0
    for path in sdf_paths:
        molecules = list(Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False))
        molecule_count += len(molecules)
        invalid_count += sum(molecule is None for molecule in molecules)
    if len(sdf_paths) != 21 or invalid_count:
        raise ValueError(
            f"SDF validation failed: files={len(sdf_paths)}, invalid={invalid_count}"
        )
    return f"RDKit parsed {molecule_count} molecules across {len(sdf_paths)} SDF files"


def main() -> int:
    checks: list[tuple[str, Any]] = [
        ("JSON documents", verify_json),
        ("source/artifact/result manifests", verify_manifests),
        ("partial evaluation scope", verify_results),
        ("checkpoint structure", verify_checkpoint),
    ]
    try:
        files = tracked_files()
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise ValueError(f"tracked files missing from worktree: {missing}")
        name_violations = scan_forbidden_names(ROOT, files)
        content_violations = scan_forbidden_content(ROOT, files)
        if name_violations or content_violations:
            raise ValueError("; ".join(name_violations + content_violations))
        print(f"PASS tracked-file security scan ({len(files)} files)")

        for label, check in checks:
            detail = check()
            suffix = f" ({detail})" if detail is not None else ""
            print(f"PASS {label}{suffix}")

        verify_lfs(files)
        print("PASS Git LFS and large-file policy")
        print(f"PASS SDF validation ({verify_sdf()})")
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print("Repository verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
