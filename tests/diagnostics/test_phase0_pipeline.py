import tempfile
import unittest
from pathlib import Path

import numpy as np

from dfe.diagnostics.contracts import canonical_sha256, write_new_manifest
from dfe.diagnostics.io import sha256_file
from dfe.diagnostics.ledger import AttemptLedger, replay_ledger
from dfe.diagnostics.metrics import summarize_attempts, terminal_attempt_records
from dfe.diagnostics.openness import compute_openness
from dfe.diagnostics.statistics import smoke_gate


class SyntheticPhase0PipelineTests(unittest.TestCase):
    def test_hash_linked_pipeline_preserves_failures_and_is_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "schema_version": "phase0.v1",
                "run_id": "synthetic",
                "manifest_hash": "",
            }
            manifest["manifest_hash"] = canonical_sha256(manifest)
            manifest_path = root / "run-manifest.json"
            write_new_manifest(manifest_path, manifest)
            original_hash = sha256_file(manifest_path)

            directions = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
            closed = compute_openness(directions, ["C", "C"], np.zeros(3), direction_count=64)
            opened = compute_openness(directions[:1], ["C"], np.zeros(3), direction_count=64)
            self.assertGreater(opened.openness, closed.openness)

            attempts_path = root / "attempts.jsonl"
            with AttemptLedger(attempts_path) as ledger:
                for index, success in enumerate((True, False, True, False)):
                    attempt_id = f"synthetic:p0:20260901:D0:{index:04d}"
                    ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "requested"})
                    ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "initialized"})
                    ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "sampling"})
                    if success:
                        ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "generated"})
                        ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "reconstructed"})
                        ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "evaluated", "smiles": f"C{index}"})
                    else:
                        ledger.append({"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": "failed", "error_code": "queue_empty"})
            terminal = terminal_attempt_records(replay_ledger(attempts_path).records)
            summary = summarize_attempts(terminal)
            self.assertEqual(summary["validity"]["rate"], 0.5)
            gate = smoke_gate(
                {
                    "expected_job_count": 1,
                    "terminal_job_count": 1,
                    "expected_attempts_per_job": 10,
                    "jobs_with_exact_attempt_count": 1,
                    "normal_parity_passed": True,
                    "finite_traces": True,
                    "clean_replay": True,
                    "output_hashes_valid": True,
                },
                retained_arm_ids=["D0", "D2"],
            )
            self.assertEqual(gate["status"], "pass")
            with self.assertRaises(FileExistsError):
                write_new_manifest(manifest_path, manifest)
            self.assertEqual(sha256_file(manifest_path), original_hash)


if __name__ == "__main__":
    unittest.main()
