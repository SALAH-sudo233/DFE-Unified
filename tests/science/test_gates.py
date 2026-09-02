import unittest

from dfe.science.gates import evaluate_gate, retry_action


class ScienceGateTests(unittest.TestCase):
    def test_scientific_failure_requires_research_before_retry(self):
        result = evaluate_gate("SCI-1-SE3-v1", evidence_complete=True, thresholds_pass=False)
        self.assertEqual(result["status"], "scientific_fail")
        self.assertEqual(retry_action(result), "research_and_targeted_fix")

    def test_infrastructure_block_is_not_scientific_failure(self):
        result = evaluate_gate("SCI-2A-FEATURE-v1", evidence_complete=False, thresholds_pass=False)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(retry_action(result), "repair_inputs_or_environment")

    def test_pass_allows_next_stage(self):
        result = evaluate_gate("SCI-2A-FEATURE-v1", evidence_complete=True, thresholds_pass=True)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(retry_action(result), "advance")
