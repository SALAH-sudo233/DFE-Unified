import unittest

from dfe.diagnostics.metrics import summarize_attempts


class MetricsTests(unittest.TestCase):
    def test_end_to_end_denominators_include_all_requested_attempts(self):
        attempts = []
        for index in range(10):
            record = {"attempt_id": f"a{index}", "requested": True}
            if index < 2:
                record.update(status="failed", error_code="queue_empty")
            elif index == 2:
                record.update(status="failed", error_code="reconstruction_error")
            elif index == 3:
                record.update(status="failed", error_code="disconnected")
            else:
                record.update(
                    status="evaluated",
                    smiles="CC" if index < 6 else f"C{'N' * index}",
                    docking_score=-7.0 if index < 8 else None,
                    posebusters_pass=index in (4, 5, 6),
                )
            attempts.append(record)
        summary = summarize_attempts(attempts)
        self.assertEqual(summary["validity"], {"numerator": 6, "denominator": 10, "rate": 0.6})
        self.assertEqual(summary["dockable"]["denominator"], 10)
        self.assertEqual(summary["dockable"]["numerator"], 4)
        self.assertEqual(summary["dockable"]["computable"], 4)
        self.assertEqual(summary["posebusters"]["denominator"], 10)
        self.assertEqual(summary["posebusters"]["numerator"], 3)
        self.assertEqual(summary["posebusters"]["computable"], 6)
        self.assertEqual(summary["uniqueness"]["denominator"], 6)
        self.assertEqual(summary["uniqueness"]["numerator"], 5)

    def test_failure_taxonomy_uses_first_failure_and_stable_vocabulary(self):
        attempts = [
            {"attempt_id": "a0", "status": "failed", "error_code": "queue_empty"},
            {"attempt_id": "a1", "status": "evaluated", "smiles": "CC"},
        ]
        summary = summarize_attempts(attempts)
        self.assertEqual(summary["failure_taxonomy"], {"queue_empty": 1, "success": 1})
        with self.assertRaisesRegex(ValueError, "unknown failure code"):
            summarize_attempts([{"attempt_id": "bad", "status": "failed", "error_code": "mystery"}])

    def test_d5_gate_variants_remain_distinct_group_keys(self):
        from dfe.diagnostics.metrics import aggregation_key

        low = {"pocket_id": "10gs", "seed": 1, "arm_id": "D5-g0.25"}
        high = {"pocket_id": "10gs", "seed": 1, "arm_id": "D5-g1.5"}
        self.assertNotEqual(aggregation_key(low), aggregation_key(high))


if __name__ == "__main__":
    unittest.main()
