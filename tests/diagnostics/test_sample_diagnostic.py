import json
import tempfile
import unittest
from pathlib import Path

from dfe.diagnostics.ledger import AttemptLedger, replay_ledger
from sample_diagnostic import (
    load_declared_job,
    predeclare_attempts,
    recover_interrupted_attempt,
)


class DiagnosticSamplerContractTests(unittest.TestCase):
    def test_smoke_job_predeclares_exactly_ten_attempts(self):
        job = {
            "job_id": "smoke:10gs:20260901:D0",
            "stage": "smoke",
            "pocket_id": "10gs",
            "seed": 20260901,
            "intervention": "D0",
            "arm_id": "D0",
            "gate": 1.0,
            "attempt_count": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            predeclare_attempts(path, "run", job)
            replay = replay_ledger(path)
        self.assertEqual(len(replay.states), 10)
        self.assertEqual(set(replay.states.values()), {"requested"})

    def test_job_must_be_declared_by_manifest_and_output_is_derived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs.jsonl"
            job = {
                "job_id": "main:10gs:20260901:D0",
                "stage": "main",
                "pocket_id": "10gs",
                "seed": 20260901,
                "intervention": "D0",
                "arm_id": "D0",
                "gate": 1.0,
                "attempt_count": 20,
            }
            jobs.write_text(json.dumps(job) + "\n", encoding="ascii")
            manifest = {"artifacts": {"jobs": {"path": "jobs.jsonl"}}}
            loaded, output = load_declared_job(root, manifest, job["job_id"])
            self.assertEqual(loaded, job)
            self.assertEqual(output, root / "jobs" / job["job_id"])
            with self.assertRaisesRegex(ValueError, "not declared"):
                load_declared_job(root, manifest, "main:xxxx:20260901:D0")

    def test_resume_closes_generated_and_reconstructed_attempts_without_backtracking(self):
        job = {
            "job_id": "main:10gs:20260901:D0",
            "pocket_id": "10gs",
            "seed": 20260901,
            "arm_id": "D0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "attempts.jsonl"
            with AttemptLedger(path) as ledger:
                for index, final_status in ((0, "generated"), (1, "reconstructed")):
                    attempt_id = f"run:10gs:20260901:D0:{index:04d}"
                    for status in ("requested", "initialized", "sampling", "generated"):
                        ledger.append(
                            {"schema_version": "phase0.v1", "attempt_id": attempt_id, "status": status}
                        )
                    if final_status == "reconstructed":
                        ledger.append(
                            {
                                "schema_version": "phase0.v1",
                                "attempt_id": attempt_id,
                                "status": "reconstructed",
                            }
                        )
            with AttemptLedger(path, resume=True) as ledger:
                recover_interrupted_attempt(ledger, "run", job, 0, root)
                recover_interrupted_attempt(ledger, "run", job, 1, root)
                self.assertEqual(ledger.states["run:10gs:20260901:D0:0000"], "failed")
                self.assertEqual(ledger.states["run:10gs:20260901:D0:0001"], "evaluated")


if __name__ == "__main__":
    unittest.main()
