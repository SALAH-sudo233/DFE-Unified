import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dfe.diagnostics.ledger import AttemptLedger, replay_ledger
from dfe.diagnostics.sampling import relax_initialization_thresholds
from sample_diagnostic import (
    _attempt_payload,
    _write_sdf,
    classify_initial_queue,
    close_requested_after_bootstrap_failure,
    load_declared_job,
    predeclare_attempts,
    recover_interrupted_attempt,
    parity_projection,
)


class DiagnosticSamplerContractTests(unittest.TestCase):
    def test_parity_projection_excludes_molecular_content_and_paths(self):
        record = {
            "status": "evaluated",
            "error_code": None,
            "smiles": "CCO",
            "candidate_path": "secret/candidate.pt",
            "sdf_path": "secret/candidate.sdf",
        }
        projection = parity_projection(record)
        self.assertEqual(
            projection,
            {
                "status": "evaluated",
                "error_code": None,
                "has_smiles": True,
            },
        )
        self.assertNotIn("smiles", projection)

    def test_sdf_writer_rejects_missing_or_empty_output(self):
        class NoOutputChem:
            @staticmethod
            def MolToMolFile(molecule, path):
                del molecule, path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.sdf"
            with self.assertRaisesRegex(OSError, "non-empty"):
                _write_sdf(NoOutputChem, object(), path)

    def test_initialization_threshold_exhaustion_is_terminal(self):
        threshold = SimpleNamespace(
            pos_threshold=0.1,
            focal_threshold=0.2,
            element_threshold=0.3,
        )
        changed = relax_initialization_thresholds(
            threshold,
            pdf_pos=torch.tensor([0.2]),
            p_focal=torch.tensor([0.3]),
            element_prob=torch.tensor([0.4]),
        )
        self.assertFalse(changed)
        self.assertEqual(threshold.pos_threshold, 0.1)
        self.assertEqual(threshold.focal_threshold, 0.2)
        self.assertEqual(threshold.element_threshold, 0.3)

    def test_empty_finished_initial_candidate_is_init_no_frontier(self):
        class Candidate:
            status = "finished"
            ligand_context_pos = []

        self.assertEqual(
            classify_initial_queue([Candidate()], "finished"),
            "init_no_frontier",
        )

    def test_bootstrap_failure_closes_every_requested_attempt(self):
        job = {
            "job_id": "smoke:10gs:20260901:D0",
            "stage": "smoke",
            "pocket_id": "10gs",
            "seed": 20260901,
            "intervention": "D0",
            "arm_id": "D0",
            "attempt_count": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            predeclare_attempts(path, "run", job)
            close_requested_after_bootstrap_failure(path, "run", job, "missing dependency")
            replay = replay_ledger(path)
        self.assertEqual(len(replay.states), 10)
        self.assertEqual(set(replay.states.values()), {"failed"})
        terminal = [record for record in replay.records if record["status"] == "failed"]
        self.assertEqual({record["error_code"] for record in terminal}, {"runtime_error"})

    def test_attempt_payload_preserves_arm_id_for_d5_gate_variants(self):
        job = {
            "job_id": "main:10gs:20260901:D5-g0.25",
            "pocket_id": "10gs",
            "seed": 20260901,
            "intervention": "D5",
            "arm_id": "D5-g0.25",
        }
        payload = _attempt_payload("run", job, 0, "requested")
        self.assertEqual(payload["intervention"], "D5")
        self.assertEqual(payload["arm_id"], "D5-g0.25")

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
            "intervention": "D0",
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
            terminal = {
                record["attempt_id"]: record
                for record in replay_ledger(path).records
                if record["status"] in {"evaluated", "failed"}
            }
            self.assertEqual(
                {record["error_code"] for record in terminal.values()},
                {"runtime_error"},
            )


if __name__ == "__main__":
    unittest.main()
