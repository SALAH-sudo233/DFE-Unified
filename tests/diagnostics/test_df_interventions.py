import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

from dfe.diagnostics.interventions import (
    DFIntervention,
    compute_df_with_diagnostics,
)
from models.df_module import AnalyticalDirectionField


ROOT = Path(__file__).resolve().parents[2]


def synthetic_inputs():
    query = torch.tensor([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]])
    pocket = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-1.0, 0.0, 4.0]])
    types = torch.tensor([0, 2, 4])
    mask = torch.tensor([True, True, False])
    return query, pocket, types, mask


class DFSeparationTests(unittest.TestCase):
    def test_raw_project_separation_is_bitwise_equal_to_legacy_forward(self):
        torch.manual_seed(7)
        module = AnalyticalDirectionField(hidden_dim=16)
        inputs = synthetic_inputs()
        legacy = module(*inputs).detach().clone()
        raw = module.raw_features(*inputs)
        current = module.project_features(raw)
        torch.testing.assert_close(current, legacy, rtol=0, atol=0)
        self.assertEqual(raw.shape, (2, 8))

    def test_df_checkpoint_keys_are_unchanged(self):
        easydict = types.ModuleType("easydict")

        class EasyDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

        EasyDict.__module__ = "easydict"
        EasyDict.__qualname__ = "EasyDict"
        easydict.EasyDict = EasyDict
        sys.modules.setdefault("easydict", easydict)
        checkpoint = torch.load(
            ROOT / "artifacts" / "checkpoints" / "df-500k.pt",
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_keys = {
            key.removeprefix("df_module.")
            for key in checkpoint["model"]
            if key.startswith("df_module.")
        }
        module_keys = set(AnalyticalDirectionField(hidden_dim=64).state_dict())
        self.assertEqual(module_keys, checkpoint_keys)
        self.assertEqual(len(checkpoint["model"]), 392)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.raw = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    def test_d1_zeroes_projected_features(self):
        output = DFIntervention.from_arm("D1").apply_projected(torch.ones(4, 3))
        torch.testing.assert_close(output, torch.zeros(4, 3), rtol=0, atol=0)

    def test_d2_zeroes_only_raw_direction_columns(self):
        output = DFIntervention.from_arm("D2").apply_raw(self.raw)
        torch.testing.assert_close(output[:, 1:4], torch.zeros(4, 3), rtol=0, atol=0)
        torch.testing.assert_close(output[:, [0, 4, 5, 6, 7]], self.raw[:, [0, 4, 5, 6, 7]])

    def test_d3_shuffle_is_deterministic_and_preserves_rows(self):
        first = DFIntervention.from_arm("D3", shuffle_seed=42).apply_raw(self.raw)
        second = DFIntervention.from_arm("D3", shuffle_seed=42).apply_raw(self.raw)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual({tuple(row.tolist()) for row in first}, {tuple(row.tolist()) for row in self.raw})
        self.assertFalse(torch.equal(first, self.raw))

    def test_d4_replaces_raw_features_and_checks_shape(self):
        alternate = torch.full_like(self.raw, 9.0)
        output = DFIntervention.from_arm("D4", alternate_raw=alternate).apply_raw(self.raw)
        torch.testing.assert_close(output, alternate, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            DFIntervention.from_arm("D4", alternate_raw=alternate[:2]).apply_raw(self.raw)

    def test_d5_accepts_only_frozen_gate_values(self):
        for gate in (0.25, 0.5, 1.0, 1.5):
            intervention = DFIntervention.from_arm("D5", gate=gate)
            torch.testing.assert_close(
                intervention.apply_projected(torch.ones(2, 2)),
                torch.full((2, 2), gate),
            )
        with self.assertRaisesRegex(ValueError, "gate"):
            DFIntervention.from_arm("D5", gate=0.75)
        with self.assertRaisesRegex(ValueError, "unknown"):
            DFIntervention.from_arm("D9")

    def test_model_entry_preserves_default_path_and_observes_copies(self):
        torch.manual_seed(5)
        module = AnalyticalDirectionField(hidden_dim=8)
        final_projection = nn.Linear(8, 6)
        inputs = synthetic_inputs()
        expected = final_projection(module(*inputs))
        actual = compute_df_with_diagnostics(module, final_projection, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        observed = {}

        def observer(name, value):
            observed[name] = value

        diagnostic = compute_df_with_diagnostics(
            module,
            final_projection,
            inputs,
            intervention=DFIntervention.from_arm("D0"),
            observer=observer,
        )
        torch.testing.assert_close(diagnostic, expected, rtol=0, atol=0)
        self.assertEqual(set(observed), {"df.raw", "df.hidden", "df.projected"})
        observed["df.projected"].zero_()
        self.assertFalse(torch.equal(diagnostic, observed["df.projected"]))


if __name__ == "__main__":
    unittest.main()
