"""Protein-conditioned residual for edge attention.

Zero-initialized so enabling it preserves the pretrained generator at step 0.
Inputs are invariant scalar edge features; relative-vector channels are expected
 to be encoded upstream with rotation-safe operations.
"""
import torch
from torch import nn


class InteractionResidual(nn.Module):
    """Predict an additive scalar edge-attention residual."""

    def __init__(self, in_features, hidden_features, out_features, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if not self.enabled:
            return torch.zeros(*features.shape[:-1], self.net[-1].out_features,
                               device=features.device, dtype=features.dtype)
        return self.net(features)
