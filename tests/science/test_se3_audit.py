import unittest
import numpy as np

from dfe.science.se3_audit import (
    compare_invariant,
    compare_position,
    compare_vector,
    first_discrete_divergence,
)


class SE3AuditTests(unittest.TestCase):
    def test_rotation_and_translation_laws(self):
        x = np.array([[1.0, 2.0, 3.0]])
        r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([4.0, -2.0, 1.0])
        self.assertLess(compare_vector(x, x @ r.T, r).normalized_max, 1e-8)
        self.assertLess(compare_position(x, x @ r.T + t, r, t).normalized_max, 1e-8)

    def test_first_divergence_is_stable(self):
        self.assertEqual(first_discrete_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertIsNone(first_discrete_divergence([1, 2], [1, 2]))
