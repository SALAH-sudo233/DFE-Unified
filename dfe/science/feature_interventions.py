"""Column-level interventions for frozen-checkpoint DF feature attribution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


FEATURE_NAMES = (
    "nearest_distance",
    "direction_x",
    "direction_y",
    "direction_z",
    "charge_proxy",
    "hydrophobicity_proxy",
    "distance_sq",
    "inverse_distance",
)


@dataclass(frozen=True)
class FeatureIntervention:
    name: str
    zero_columns: tuple[int, ...] = ()
    shuffle_direction: bool = False

    def apply(self, raw_features: Tensor, *, seed: int) -> Tensor:
        if raw_features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("DF raw features must have exactly eight columns")
        value = raw_features.clone()
        if self.zero_columns:
            value[..., self.zero_columns] = 0
        if self.shuffle_direction:
            rows = value.reshape(-1, value.shape[-1])
            generator = torch.Generator(device=value.device)
            generator.manual_seed(int(seed))
            permutation = torch.randperm(rows.shape[0], generator=generator, device=value.device)
            rows[:, 1:4] = rows[permutation, 1:4].clone()
        return value

    def provenance(self) -> dict[str, object]:
        return {
            "name": self.name,
            "zero_features": [FEATURE_NAMES[index] for index in self.zero_columns],
            "shuffle_direction": self.shuffle_direction,
        }

    def apply_projected(self, projected: Tensor) -> Tensor:
        """Keep projected hidden features unchanged; SCI-2A edits raw columns only."""
        return projected


INTERVENTIONS = (
    FeatureIntervention("full"),
    FeatureIntervention("direction_zero", zero_columns=(1, 2, 3)),
    FeatureIntervention("direction_shuffle", shuffle_direction=True),
    FeatureIntervention("no_distance_sq", zero_columns=(6,)),
    FeatureIntervention("no_inverse_distance", zero_columns=(7,)),
    FeatureIntervention("no_charge", zero_columns=(4,)),
    FeatureIntervention("no_hydrophobicity", zero_columns=(5,)),
    FeatureIntervention("attributes_zero", zero_columns=(1, 2, 3, 4, 5)),
)


def get_intervention(name: str) -> FeatureIntervention:
    for intervention in INTERVENTIONS:
        if intervention.name == name:
            return intervention
    raise ValueError(f"unknown SCI-2A intervention: {name}")
