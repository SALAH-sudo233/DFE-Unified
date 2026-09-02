import unittest
from types import SimpleNamespace

import torch

from dfe.science.feature_interventions import get_intervention
from models.df_module import AnalyticalDirectionField


class ScienceHookTests(unittest.TestCase):
    def test_raw_feature_intervention_is_column_scoped(self):
        module = AnalyticalDirectionField(hidden_dim=4)
        query = torch.tensor([[0.0, 0.0, 0.0]])
        pocket = torch.tensor([[1.0, 0.0, 0.0]])
        raw = module.raw_features(query, pocket, torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool))
        changed = get_intervention("no_distance_sq").apply(raw, seed=0)
        self.assertTrue(torch.equal(raw[..., :6], changed[..., :6]))
        self.assertTrue(torch.equal(changed[..., 6], torch.zeros_like(changed[..., 6])))

    def test_intervention_has_projection_identity_for_model_hook(self):
        intervention = get_intervention("full")
        values = torch.randn(2, 4)
        self.assertTrue(torch.equal(intervention.apply_projected(values), values))
