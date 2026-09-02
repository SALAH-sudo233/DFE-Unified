"""Read-only validation for science analyses consuming Phase 0 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dfe.diagnostics.contracts import PHASE0_CHECKPOINT_SHA256, canonical_sha256
from dfe.diagnostics.io import sha256_file

from . import SCIENCE_EXPERIMENT_IDS


@dataclass(frozen=True)
class ScienceManifest:
    path: Path
    payload: Mapping[str, Any]

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.payload.get("inputs", {}).get("checkpoint", {}).get("sha256", ""))

    @property
    def manifest_sha256(self) -> str:
        return str(self.payload.get("manifest_hash", ""))

    def require(self, experiment_id: str, checkpoint_sha256: str = PHASE0_CHECKPOINT_SHA256) -> None:
        if experiment_id not in SCIENCE_EXPERIMENT_IDS:
            raise ValueError(f"unknown science experiment ID: {experiment_id}")
        if self.payload.get("schema_version") != "phase0.v1":
            raise ValueError("science inputs require a phase0.v1 manifest")
        if self.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("checkpoint hash does not match the required science anchor")
        recorded = self.manifest_sha256
        if recorded:
            expected = canonical_sha256({**self.payload, "manifest_hash": ""})
            if recorded != expected:
                raise ValueError("Phase 0 manifest hash is invalid")


def load_science_manifest(path: Path) -> ScienceManifest:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 0 manifest must be a JSON object")
    return ScienceManifest(path=path, payload=payload)


def verify_artifact_hash(path: Path, expected_sha256: str) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"artifact hash mismatch: {path}")


def assert_create_only_output(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(path)
