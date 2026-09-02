import unittest

from scripts.run_d0_parity import compare_parity_runs


class D0ParityTests(unittest.TestCase):
    def test_parity_compares_terminal_semantics_smiles_and_decision_hash(self):
        baseline = {
            "projection": {"status": "evaluated", "error_code": None, "has_smiles": True},
            "smiles": "CC",
            "decision_hash": "abc",
            "decision_count": 4,
        }
        passed = compare_parity_runs(baseline, dict(baseline))
        self.assertEqual(passed["status"], "pass")
        self.assertNotIn("smiles", passed)
        changed = dict(baseline, decision_hash="different")
        failed = compare_parity_runs(baseline, changed)
        self.assertEqual(failed["status"], "fail")
        self.assertFalse(failed["checks"]["decision_trace"])


if __name__ == "__main__":
    unittest.main()
