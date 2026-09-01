"""
Direction Field module for Pocket2Mol.

Computes analytical geometric fields from pocket structure:
  1. Distance to nearest pocket atom (scalar)
  2. Direction vector to nearest pocket atom (3D vector)
  3. Electrostatic potential from partial charges
  4. Hydrophobicity-weighted distance
  5. Distance squared
  6. Inverse distance

These fields are deterministic functions of pocket structure,
providing geometric priors for atom placement.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AnalyticalDirectionField(nn.Module):
    """Compute geometric fields analytically from pocket atoms."""

    PARTIAL_CHARGES = torch.tensor([0.1, -0.3, -0.4, 0.2, -0.2])
    HYDROPHOBICITY = torch.tensor([1.5, -3.0, -3.5, -1.0, -1.5])

    def __init__(self, hidden_dim: int = 256, num_protein_types: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_protein_types = num_protein_types
        input_dim = 8

        self.field_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.register_buffer(
            "partial_charges",
            self.PARTIAL_CHARGES[:num_protein_types].clone(),
        )
        self.register_buffer(
            "hydrophobicity",
            self.HYDROPHOBICITY[:num_protein_types].clone(),
        )

    def forward(
        self,
        query_points: Tensor,   # (Q, 3)
        pocket_pos: Tensor,     # (P, 3)
        pocket_types: Tensor,   # (P,)
        pocket_mask: Tensor,    # (P,) bool
    ) -> Tensor:
        """Returns: (Q, hidden_dim) field features at each query point."""
        return self.project_features(
            self.raw_features(query_points, pocket_pos, pocket_types, pocket_mask)
        )

    def raw_features(
        self,
        query_points: Tensor,
        pocket_pos: Tensor,
        pocket_types: Tensor,
        pocket_mask: Tensor,
    ) -> Tensor:
        """Return the existing eight analytical inputs before scalar projection."""
        diff = pocket_pos.unsqueeze(1) - query_points.unsqueeze(0)  # (P, Q, 3)
        dist = (diff ** 2).sum(dim=-1).add(1e-8).sqrt()  # (P, Q)

        dist_masked = dist.masked_fill(~pocket_mask.unsqueeze(1), float("inf"))
        min_dist, min_idx = dist_masked.min(dim=0)  # (Q,), (Q,)

        nearest_diff = diff.gather(
            0, min_idx.unsqueeze(0).unsqueeze(-1).expand(1, query_points.size(0), 3)
        ).squeeze(0)  # (Q, 3)
        dir_vec = nearest_diff / (min_dist.unsqueeze(-1) + 1e-8)

        charges = self.partial_charges[pocket_types.clamp(0, self.num_protein_types - 1)]
        coulomb = charges.unsqueeze(1) / (dist + 1.0)
        coulomb = coulomb.masked_fill(~pocket_mask.unsqueeze(1), 0.0)
        electrostatic = coulomb.sum(dim=0)  # (Q,)

        hydro = self.hydrophobicity[pocket_types.clamp(0, self.num_protein_types - 1)]
        hydro_weight = hydro.unsqueeze(1) * pocket_mask.unsqueeze(1).float()
        hydro_dist = hydro_weight / (dist + 1.0)
        hydrophobic_field = hydro_dist.sum(dim=0)  # (Q,)

        dist_sq = min_dist ** 2
        inv_dist = 1.0 / (min_dist + 1.0)

        return torch.cat([
            min_dist.unsqueeze(-1),
            dir_vec,
            electrostatic.unsqueeze(-1),
            hydrophobic_field.unsqueeze(-1),
            dist_sq.unsqueeze(-1),
            inv_dist.unsqueeze(-1),
        ], dim=-1)  # (Q, 8)

    def project_features(self, raw_features: Tensor) -> Tensor:
        """Project raw analytical inputs with the unchanged checkpoint module."""
        return self.field_proj(raw_features)  # (Q, hidden_dim)
