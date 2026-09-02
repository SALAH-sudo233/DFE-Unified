import unittest
import torch

from dfe.science.feature_interventions import INTERVENTIONS


class FeatureInterventionTests(unittest.TestCase):
    def test_matrix_contains_frozen_feature_arms(self):
        names = {item.name for item in INTERVENTIONS}
        self.assertEqual(names, {"full", "direction_zero", "direction_shuffle", "no_distance_sq", "no_inverse_distance", "no_charge", "no_hydrophobicity", "attributes_zero"})

    def test_interventions_do_not_mutate_input_and_are_seeded(self):
        raw = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        original = raw.clone()
        shuffled = next(item for item in INTERVENTIONS if item.name == "direction_shuffle")
        self.assertTrue(torch.equal(shuffled.apply(raw, seed=7), shuffled.apply(raw, seed=7)))
        self.assertTrue(torch.equal(raw, original))
