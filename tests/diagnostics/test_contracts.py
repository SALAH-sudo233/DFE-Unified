import json
import tempfile
import unittest
from pathlib import Path

from dfe.diagnostics.contracts import (
    AttemptRecord,
    JobSpec,
    PocketRecord,
    RunManifest,
    canonical_sha256,
    load_phase0_config,
    write_new_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "diagnostics" / "phase0_df500k.yaml"
CHECKPOINT_SHA256 = (
    "34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0"
)


class ContractTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self):
        self.assertEqual(
            canonical_sha256({"a": 1, "b": 2}),
            canonical_sha256({"b": 2, "a": 1}),
        )

    def test_manifest_is_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_new_manifest(path, {"experiment_id": "P0-MANIFEST-v1"})
            self.assertEqual(
                json.loads(path.read_text(encoding="ascii")),
                {"experiment_id": "P0-MANIFEST-v1"},
            )
            with self.assertRaises(FileExistsError):
                write_new_manifest(path, {"experiment_id": "changed"})

    def test_attempt_requires_all_attempt_denominator_fields(self):
        record = AttemptRecord.new("run", "10gs", 20260901, "D0", 0)
        payload = record.to_dict()
        self.assertEqual(payload["status"], "requested")
        self.assertEqual(payload["attempt_id"], "run:10gs:20260901:D0:0000")
        self.assertIn("sampling_status", payload)
        self.assertIn("reconstruction_status", payload)
        self.assertIn("evaluation_status", payload)

    def test_phase0_config_freezes_anchor_and_protocol(self):
        config = load_phase0_config(CONFIG)
        self.assertEqual(config.schema_version, "phase0.v1")
        self.assertEqual(config.checkpoint.sha256, CHECKPOINT_SHA256)
        self.assertEqual(config.smoke_attempts, 10)
        self.assertEqual(config.main_attempts, 20)
        self.assertEqual(config.seeds, (20260901, 20260902, 20260903))
        self.assertEqual(config.openness.direction_count, 2048)
        self.assertEqual(config.openness.cutoff_angstrom, 12.0)
        self.assertEqual(config.interventions, ("D0", "D1", "D2", "D3", "D4", "D5"))

    def test_phase0_config_rejects_unknown_keys_and_checkpoint_mismatch(self):
        config = CONFIG.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase0.yaml"
            path.write_text(config + "unknown_key: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                load_phase0_config(path)

            path.write_text(config.replace(CHECKPOINT_SHA256, "0" * 64), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                load_phase0_config(path)

    def test_manifest_rejects_duplicate_ids_and_invalid_jobs(self):
        pocket = PocketRecord("10gs", {"x": 1.0, "y": 2.0, "z": 3.0}, ())
        with self.assertRaisesRegex(ValueError, "duplicate pocket IDs"):
            RunManifest.new("run", CHECKPOINT_SHA256, (pocket, pocket), ())

        with self.assertRaisesRegex(ValueError, "smoke attempt count"):
            JobSpec("job", "smoke", "10gs", 20260901, "D0", 20)
        with self.assertRaisesRegex(ValueError, "frozen Phase 0 seeds"):
            JobSpec("job", "main", "10gs", 7, "D0", 20)

        job = JobSpec("job", "main", "10gs", 20260901, "D0", 20)
        with self.assertRaisesRegex(ValueError, "duplicate job keys"):
            RunManifest.new("run", CHECKPOINT_SHA256, (pocket,), (job, job))


if __name__ == "__main__":
    unittest.main()
