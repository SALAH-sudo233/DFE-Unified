import unittest

import numpy as np

from dfe.diagnostics.se3 import (
    apply_points,
    apply_vectors,
    compare_equivariant,
    compare_invariant,
    normalized_error,
    sample_so3,
)


class SE3PrimitiveTests(unittest.TestCase):
    def test_sample_so3_is_deterministic_orthogonal_and_proper(self):
        rotations = sample_so3(20260901, 100)
        np.testing.assert_array_equal(rotations, sample_so3(20260901, 100))
        identity = np.eye(3)
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, identity, atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=14)

    def test_apply_points_preserves_distances_and_adds_translation(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
        rotation = sample_so3(7, 1)[0]
        moved = apply_points(points, rotation, np.array([5.0, -1.0, 2.0]))
        before = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        after = np.linalg.norm(moved[:, None] - moved[None, :], axis=-1)
        np.testing.assert_allclose(after, before, atol=1e-14)
        np.testing.assert_allclose(moved[0], [5.0, -1.0, 2.0], atol=1e-14)

    def test_vectors_rotate_without_translation_and_compare_after_alignment(self):
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]])
        rotation = sample_so3(9, 1)[0]
        transformed = apply_vectors(vectors, rotation)
        stats = compare_equivariant(transformed, vectors, rotation)
        self.assertLess(stats.normalized_max, 1e-12)
        np.testing.assert_allclose(
            apply_vectors(vectors, rotation, translation=np.array([100.0, 100.0, 100.0])),
            transformed,
        )

    def test_invariant_comparison_rejects_xyz_components(self):
        expected = np.array([[1.0, 2.0, 3.0]])
        actual = apply_vectors(expected, sample_so3(11, 1)[0])
        stats = compare_invariant(actual, expected)
        self.assertGreater(stats.normalized_max, 0.1)

    def test_zero_reference_uses_stable_denominator_and_absolute_context(self):
        stats = normalized_error(np.array([[1e-13, 0.0, 0.0]]), np.zeros((1, 3)))
        self.assertAlmostEqual(stats.normalized_max, 0.1)
        self.assertEqual(stats.absolute_max, 1e-13)
        zero = normalized_error(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertEqual(zero.normalized_max, 0.0)
        self.assertEqual(zero.absolute_p95, 0.0)

    def test_multichannel_outputs_use_row_norm_not_elementwise_relative_error(self):
        expected = np.array([[1.0, 0.0, 0.0, 1.0]])
        actual = expected + np.array([[0.0, 1e-13, 0.0, 0.0]])
        stats = normalized_error(actual, expected)
        self.assertLess(stats.normalized_max, 1e-12)


if __name__ == "__main__":
    unittest.main()
