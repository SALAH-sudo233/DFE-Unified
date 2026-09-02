import json
import tempfile
import unittest
from pathlib import Path

from dfe.diagnostics.contracts import canonical_json
from dfe.diagnostics.ledger import AttemptLedger
from scripts.analyze_phase0 import _verify_output_hashes, complete_arm_ids


class AnalyzePhase0Tests(unittest.TestCase):
    def test_successful_attempt_requires_candidate_and_sdf_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(
                _verify_output_hashes(
                    root,
                    [{"status": "evaluated", "smiles": "CC"}],
                )
            )
            self.assertTrue(
                _verify_output_hashes(
                    root,
                    [{"status": "failed", "error_code": "queue_empty"}],
                )
            )

    def test_optional_arm_requires_all_jobs_to_be_terminal(self):
        jobs = [
            {
                "job_id": f"smoke-p{index}-20260901-D3",
                "arm_id": "D3",
                "attempt_count": 2,
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job in jobs:
                job_root = root / "jobs" / job["job_id"]
                job_root.mkdir(parents=True)
                with AttemptLedger(job_root / "attempts.jsonl") as ledger:
                    for sample_index in range(2):
                        attempt_id = f"run:{job['job_id']}:{sample_index}"
                        ledger.append(
                            {
                                "schema_version": "phase0.v1",
                                "attempt_id": attempt_id,
                                "status": "requested",
                            }
                        )
                        if job is jobs[0]:
                            ledger.append(
                                {
                                    "schema_version": "phase0.v1",
                                    "attempt_id": attempt_id,
                                    "status": "failed",
                                }
                            )
                (job_root / "events.jsonl").write_bytes(canonical_json({"event": "ok"}))
            self.assertEqual(complete_arm_ids(root, jobs), set())


if __name__ == "__main__":
    unittest.main()
