import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_sci1_se3_audit import (
    _preflight_gate,
    _preflight_transforms,
)


class SCI1CLITests(unittest.TestCase):
    def test_help_is_available_without_inputs(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_sci1_se3_audit.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--vector-origin-mode", result.stdout)
        self.assertIn("--stage", result.stdout)

    def test_invalid_vector_origin_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_sci1_se3_audit.py",
                    "--manifest",
                    "missing.json",
                    "--device",
                    "cpu",
                    "--vector-origin-mode",
                    "learned",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_missing_manifest_is_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            result = subprocess.run(
                [sys.executable, "scripts/run_sci1_se3_audit.py", "--manifest", str(Path(tmp) / "missing.json"), "--output", str(output)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "infrastructure_failure")

    def test_preflight_transforms_cover_four_rigid_categories(self):
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([4.0, -2.0, 1.0])

        transforms = _preflight_transforms(rotation, translation)

        self.assertEqual(
            [item["category"] for item in transforms],
            ["identity", "rotation", "translation", "rigid"],
        )
        np.testing.assert_array_equal(transforms[0]["rotation"], np.eye(3))
        np.testing.assert_array_equal(transforms[0]["translation"], np.zeros(3))
        np.testing.assert_array_equal(transforms[1]["rotation"], rotation)
        np.testing.assert_array_equal(transforms[1]["translation"], np.zeros(3))
        np.testing.assert_array_equal(transforms[2]["rotation"], np.eye(3))
        np.testing.assert_array_equal(transforms[2]["translation"], translation)
        np.testing.assert_array_equal(transforms[3]["rotation"], rotation)
        np.testing.assert_array_equal(transforms[3]["translation"], translation)

    def test_preflight_gate_requires_both_encoder_events_below_tolerance(self):
        records = [
            self._preflight_record(category, scalar=1e-5, vector=2e-5)
            for category in ("identity", "rotation", "translation", "rigid")
        ]

        passed, first_failure = _preflight_gate(records, tolerance=1e-4)

        self.assertTrue(passed)
        self.assertIsNone(first_failure)

        records[2]["events"][1]["normalized_max"] = 1e-4
        passed, first_failure = _preflight_gate(records, tolerance=1e-4)
        self.assertFalse(passed)
        self.assertEqual(first_failure, "translation:0:encoder.vector")

    def test_preflight_gate_rejects_topology_mismatch_first(self):
        records = [
            self._preflight_record(category, scalar=1e-5, vector=2e-5)
            for category in ("identity", "rotation", "translation", "rigid")
        ]
        records[1]["topology_match"] = False

        passed, first_failure = _preflight_gate(records, tolerance=1e-4)

        self.assertFalse(passed)
        self.assertEqual(first_failure, "rotation:topology.edge_index")

    @staticmethod
    def _preflight_record(category, scalar, vector):
        return {
            "transform_category": category,
            "topology_match": True,
            "events": [
                {
                    "key": "0:encoder.scalar",
                    "normalized_max": scalar,
                },
                {
                    "key": "0:encoder.vector",
                    "normalized_max": vector,
                },
                {
                    "key": "0:frontier.logits",
                    "normalized_max": 10.0,
                },
            ],
        }
