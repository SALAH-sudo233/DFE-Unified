import json
import tempfile
import unittest
from pathlib import Path

from dfe.science.artifact_contract import (
    assert_create_only_output,
    load_science_manifest,
)


class ArtifactContractTests(unittest.TestCase):
    def test_manifest_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"schema_version": "phase0.v1", "inputs": {"checkpoint": {"sha256": "bad"}}}), encoding="utf-8")
            manifest = load_science_manifest(path)
            with self.assertRaises(ValueError):
                manifest.require("SCI-1-SE3-v1", "wrong")

    def test_output_contract_is_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                assert_create_only_output(path)

    def test_artifact_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"stable")
            from dfe.science.artifact_contract import verify_artifact_hash
            with self.assertRaises(ValueError):
                verify_artifact_hash(path, "0" * 64)
