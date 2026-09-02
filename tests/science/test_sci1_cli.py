import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
