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
    origin = compose_pos[idx_protein].mean(dim=0, keepdim=True)
    return compose_pos - origin
