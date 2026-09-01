import unittest

import numpy as np

from dfe.diagnostics.statistics import (
    benjamini_hochberg,
    cluster_bootstrap_ci,
    fit_openness_interaction,
    phase0_gate,
    paired_effect,
    sample_pocket_clusters,
    smoke_gate,
)


def synthetic_rows():
    rows = []
    for pocket in range(30):
        openness = pocket / 29
        for seed in (20260901, 20260902, 20260903):
            rows.append(
                {
                    "pocket_id": f"p{pocket:02d}",
                    "seed": seed,
                    "intervention": "D0",
                    "openness": openness,
                    "pocket_atom_count": 100 + pocket,
                    "reference_ligand_heavy_atoms": 20 + pocket % 4,
                    "metric": 0.8 - 0.1 * openness,
                }
            )
            rows.append(
                {
                    "pocket_id": f"p{pocket:02d}",
                    "seed": seed,
                    "intervention": "D2",
                    "openness": openness,
                    "pocket_atom_count": 100 + pocket,
                    "reference_ligand_heavy_atoms": 20 + pocket % 4,
                    "metric": 0.8 - 0.1 * openness - 0.4 * openness,
                }
            )
    return rows


class StatisticsTests(unittest.TestCase):
    def test_known_negative_interaction_sign_is_recovered(self):
        fit = fit_openness_interaction(synthetic_rows(), intervention="D2")
        self.assertLess(fit.interaction, -0.35)

    def test_cluster_sampling_keeps_all_rows_for_each_drawn_pocket(self):
        rows = synthetic_rows()
        sampled = sample_pocket_clusters(rows, np.random.default_rng(7))
        counts = {}
        for record in sampled:
            counts[record["cluster_draw_id"]] = counts.get(record["cluster_draw_id"], 0) + 1
        self.assertEqual(set(counts.values()), {6})
        self.assertEqual(len(counts), 30)

    def test_identical_paired_values_have_zero_effect(self):
        baseline = {"p00": 0.5, "p01": 0.7}
        self.assertEqual(paired_effect(baseline, dict(baseline)), 0.0)

    def test_bh_fdr_is_monotone_when_sorted_by_p_value(self):
        pvalues = [0.01, 0.04, 0.03, 0.20]
        adjusted = benjamini_hochberg(pvalues)
        ordered = [adjusted[index] for index in np.argsort(pvalues)]
        self.assertEqual(ordered, sorted(ordered))
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))

    def test_fixed_seed_reproduces_exact_cluster_bootstrap_ci(self):
        rows = synthetic_rows()
        first = cluster_bootstrap_ci(
            rows,
            lambda sample: fit_openness_interaction(sample, intervention="D2").interaction,
            draws=200,
            seed=20260901,
        )
        second = cluster_bootstrap_ci(
            rows,
            lambda sample: fit_openness_interaction(sample, intervention="D2").interaction,
            draws=200,
            seed=20260901,
        )
        self.assertEqual(first, second)

    def test_smoke_gate_is_about_evidence_completeness_not_effect_direction(self):
        complete = {
            "expected_job_count": 54,
            "terminal_job_count": 54,
            "expected_attempts_per_job": 10,
            "jobs_with_exact_attempt_count": 54,
            "normal_parity_passed": True,
            "finite_traces": True,
            "clean_replay": True,
            "output_hashes_valid": True,
        }
        gate = smoke_gate(complete, retained_arm_ids=["D0", "D1", "D2"])
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["retained_arm_ids"], ["D0", "D1", "D2"])
        incomplete = dict(complete, terminal_job_count=53)
        self.assertEqual(smoke_gate(incomplete, retained_arm_ids=[])["status"], "fail")

    def test_phase0_gate_requires_every_declared_main_job(self):
        complete = {
            "expected_job_count": 270,
            "terminal_job_count": 270,
            "expected_attempts_per_job": 20,
            "jobs_with_exact_attempt_count": 270,
            "finite_traces": True,
            "clean_replay": True,
            "output_hashes_valid": True,
            "summary_artifacts_valid": True,
        }
        gate = phase0_gate(complete, se3_hypothesis="fail", analysis=[])
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["se3_hypothesis"], "fail")
        incomplete = dict(complete, terminal_job_count=269)
        self.assertEqual(
            phase0_gate(incomplete, se3_hypothesis="inconclusive", analysis=[])["status"],
            "fail",
        )
        empty = dict(complete, expected_job_count=0, terminal_job_count=0,
                     jobs_with_exact_attempt_count=0)
        self.assertEqual(
            phase0_gate(empty, se3_hypothesis="fail", analysis=[])["status"],
            "fail",
        )


if __name__ == "__main__":
    unittest.main()
