"""Science-only initial vector-origin candidates for Pocket2Mol."""

from __future__ import annotations

import torch
from torch import Tensor


VECTOR_ORIGIN_MODES = ("absolute", "centered", "zero")


def normalize_vector_origin_mode(mode: str | None) -> str:
    normalized = "absolute" if mode is None else str(mode).lower()
    if normalized not in VECTOR_ORIGIN_MODES:
        choices = ", ".join(VECTOR_ORIGIN_MODES)
        raise ValueError(
            f"invalid vector-origin mode {mode!r}; expected one of: {choices}"
        )
    return normalized


def vector_embedding_positions(
    compose_pos: Tensor,
    idx_protein: Tensor,
    mode: str | None,
    compose_batch: Tensor | None = None,
) -> Tensor:
    normalized = normalize_vector_origin_mode(mode)
    if compose_pos.ndim != 2 or compose_pos.shape[-1] != 3:
        raise ValueError("compose_pos must have shape [N, 3]")
    if normalized == "absolute":
        return compose_pos
    if normalized == "zero":
        return torch.zeros_like(compose_pos)
    if idx_protein.numel() == 0:
        raise ValueError("centered vector-origin mode requires protein atoms")
    if compose_batch is None:
        # Single-graph/inference fallback. Training passes compose_batch so
        # each pocket in a PyG batch is centered independently.
        origin = compose_pos[idx_protein].mean(dim=0, keepdim=True)
        return compose_pos - origin
    if compose_batch.ndim != 1 or compose_batch.shape[0] != compose_pos.shape[0]:
        raise ValueError("compose_batch must have shape [N]")
    graph_index = compose_batch[idx_protein].long()
    n_graphs = int(compose_batch.max().item()) + 1 if compose_batch.numel() else 0
    centers = torch.zeros((n_graphs, 3), dtype=compose_pos.dtype, device=compose_pos.device)
    counts = torch.zeros((n_graphs,), dtype=compose_pos.dtype, device=compose_pos.device)
    centers.index_add_(0, graph_index, compose_pos[idx_protein])
    counts.index_add_(0, graph_index, torch.ones_like(graph_index, dtype=compose_pos.dtype))
    if torch.any(counts <= 0):
        raise ValueError("every compose graph must contain protein atoms")
    centers = centers / counts.clamp_min(1).unsqueeze(-1)
    return compose_pos - centers[compose_batch.long()]
