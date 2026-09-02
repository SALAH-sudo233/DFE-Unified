import unittest

import numpy as np
import torch

from dfe.diagnostics.model_audit import (
    EVENT_LAWS,
    audit_analytical_df_state,
    compare_event_sets,
)
from dfe.diagnostics.observer import TensorObserver
from dfe.diagnostics.se3 import apply_points, apply_vectors, sample_so3
from models.df_module import AnalyticalDirectionField


class ModelAuditTests(unittest.TestCase):
    def setUp(self):
        self.rotation = sample_so3(17, 1)[0]
        self.translation = np.array([3.0, -2.0, 5.0])

    def test_declared_model_events_have_transformation_laws(self):
        expected = {
            "df.raw",
            "df.hidden",
            "df.projected",
            "encoder.scalar",
            "encoder.vector",
            "frontier.logits",
            "frontier.indices",
            "position.relative_mu",
            "position.absolute_mu",
            "position.sigma",
            "position.pi",
            "element.logits",
            "element.probability",
            "bond.logits",
            "bond.probability",
            "termination.has_frontier",
        }
        self.assertEqual(set(EVENT_LAWS), expected)

    def test_comparisons_apply_invariant_vector_point_and_exact_laws(self):
        reference = TensorObserver()
        transformed = TensorObserver()
        scalar = torch.tensor([[0.2, 0.8]])
        vector = np.array([[1.0, 2.0, 3.0]])
        point = np.array([[4.0, 5.0, 6.0]])
        reference.observe(0, "encoder.scalar", scalar)
        transformed.observe(0, "encoder.scalar", scalar.clone())
        reference.observe(0, "encoder.vector", torch.from_numpy(vector))
        transformed.observe(
            0, "encoder.vector", torch.from_numpy(apply_vectors(vector, self.rotation))
        )
        reference.observe(0, "position.absolute_mu", torch.from_numpy(point))
        transformed.observe(
            0,
            "position.absolute_mu",
            torch.from_numpy(apply_points(point, self.rotation, self.translation)),
        )
        reference.observe(0, "frontier.indices", torch.tensor([1, 4]))
        transformed.observe(0, "frontier.indices", torch.tensor([1, 4]))
        report = compare_event_sets(
            reference,
            transformed,
            self.rotation,
            self.translation,
            tolerance=1e-12,
        )
        self.assertTrue(report.passed)
        self.assertIsNone(report.first_failure)

    def test_first_failure_is_recorded_but_later_events_are_compared(self):
        reference = TensorObserver()
        transformed = TensorObserver()
        reference.observe(0, "frontier.logits", torch.tensor([1.0]))
        transformed.observe(0, "frontier.logits", torch.tensor([2.0]))
        reference.observe(0, "element.probability", torch.tensor([0.5]))
        transformed.observe(0, "element.probability", torch.tensor([0.7]))
        report = compare_event_sets(
            reference,
            transformed,
            self.rotation,
            self.translation,
            tolerance=1e-6,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.first_failure, "0:frontier.logits")
        self.assertEqual(len(report.events), 2)

    def test_discrete_mismatch_is_exact_failure(self):
        reference = TensorObserver()
        transformed = TensorObserver()
        reference.observe(2, "frontier.indices", torch.tensor([1, 2]))
        transformed.observe(2, "frontier.indices", torch.tensor([1, 3]))
        report = compare_event_sets(
            reference,
            transformed,
            self.rotation,
            self.translation,
            tolerance=1.0,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.events[0].law, "exact")
        self.assertEqual(report.events[0].mismatch_count, 1)

    def test_branch_divergence_records_missing_event_instead_of_raising(self):
        reference = TensorObserver()
        transformed = TensorObserver()
        reference.observe(0, "frontier.logits", torch.tensor([1.0]))
        transformed.observe(0, "frontier.logits", torch.tensor([1.0]))
        reference.observe(0, "frontier.indices", torch.tensor([0]))
        report = compare_event_sets(
            reference,
            transformed,
            self.rotation,
            self.translation,
            tolerance=1e-6,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.first_failure, "0:frontier.indices")
        self.assertEqual(report.events[-1].law, "missing_event")

    def test_numeric_shape_divergence_is_recorded_and_later_events_continue(self):
        reference = TensorObserver()
        transformed = TensorObserver()
        reference.observe(0, "encoder.vector", torch.ones(36, 3, 3))
        transformed.observe(0, "encoder.vector", torch.ones(38, 3, 3))
        reference.observe(0, "element.probability", torch.tensor([0.5]))
        transformed.observe(0, "element.probability", torch.tensor([0.5]))
        report = compare_event_sets(
            reference,
            transformed,
            self.rotation,
            self.translation,
            tolerance=1e-6,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.first_failure, "0:encoder.vector")
        self.assertEqual(report.events[0].law, "shape_mismatch")
        self.assertEqual(report.events[0].reference_shape, (36, 3, 3))
        self.assertEqual(report.events[0].transformed_shape, (38, 3, 3))
        self.assertTrue(report.events[1].passed)

    def test_analytical_df_raw_features_pass_float64_rigid_transform(self):
        module = AnalyticalDirectionField(hidden_dim=8).double()
        query = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=torch.float64)
        pocket = torch.tensor([[3.0, 0.0, 0.0], [0.0, -4.0, 1.0]], dtype=torch.float64)
        types = torch.tensor([0, 2])
        mask = torch.tensor([True, True])
        report = audit_analytical_df_state(
            module,
            query,
            pocket,
            types,
            mask,
            self.rotation,
            self.translation,
            tolerance=1e-8,
        )
        self.assertTrue(report.passed)
        self.assertLess(report.events[0].normalized_max, 1e-12)


if __name__ == "__main__":
    unittest.main()
