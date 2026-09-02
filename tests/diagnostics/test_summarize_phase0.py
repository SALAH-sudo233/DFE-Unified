import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_phase0 import selected_jobs, summary_root, verify_outputs


class SummarizePhase0Tests(unittest.TestCase):
    def test_verify_outputs_rejects_tampered_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = summary_root(root, "main")
            output.mkdir(parents=True)
            artifact_names = (
                "per-attempt.parquet",
                "per-pocket-seed.csv",
                "per-pocket.csv",
                "failure-taxonomy.csv",
            )
            from dfe.diagnostics.io import sha256_file

            for name in artifact_names:
                (output / name).write_bytes(name.encode("ascii"))
            summary = {
                "manifest_hash": "manifest",
                "metric_definition_version": "phase0-metrics.v1",
                "stage": "main",
                "artifacts": {
                    name: {
                        "size": (output / name).stat().st_size,
                        "sha256": sha256_file(output / name),
                    }
                    for name in artifact_names
                },
            }
            (output / "phase0-summary.json").write_text(
                json.dumps(summary), encoding="ascii"
            )
            verify_outputs(root, "manifest", "main")
            original = (output / "per-pocket.csv").read_bytes()
            (output / "per-pocket.csv").write_bytes(b"x" * len(original))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_outputs(root, "manifest", "main")

    def test_stage_outputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(summary_root(root, "smoke"), root / "summaries" / "smoke")
            self.assertEqual(summary_root(root, "main"), root / "summaries" / "main")

    def test_stage_selection_excludes_unrun_smoke_pockets_and_unretained_main_arms(self):
        jobs = [
            {"job_id": "smoke-p0-D0", "stage": "smoke", "pocket_id": "p0", "arm_id": "D0"},
            {"job_id": "smoke-p1-D0", "stage": "smoke", "pocket_id": "p1", "arm_id": "D0"},
            {"job_id": "main-p0-D0", "stage": "main", "pocket_id": "p0", "arm_id": "D0"},
            {"job_id": "main-p0-D3", "stage": "main", "pocket_id": "p0", "arm_id": "D3"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "smoke-pockets.json").write_text(
                json.dumps({"pocket_ids": ["p0"]}), encoding="ascii"
            )
            (root / "gate-smoke.json").write_text(
                json.dumps({"status": "pass", "retained_arm_ids": ["D0"]}),
                encoding="ascii",
            )
            self.assertEqual(
                [job["job_id"] for job in selected_jobs(root, jobs, "smoke")],
                ["smoke-p0-D0"],
            )
            self.assertEqual(
                [job["job_id"] for job in selected_jobs(root, jobs, "main")],
                ["main-p0-D0"],
            )


if __name__ == "__main__":
    unittest.main()
