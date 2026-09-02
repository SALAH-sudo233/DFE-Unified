"""Science-only inference precision selection."""

from __future__ import annotations

import torch


MODEL_DTYPES = ("float32", "float64")


def normalize_model_dtype(value: str | None) -> str:
    normalized = "float32" if value is None else str(value).lower()
    if normalized not in MODEL_DTYPES:
        choices = ", ".join(MODEL_DTYPES)
        raise ValueError(
            f"invalid model dtype {value!r}; expected one of: {choices}"
        )
    return normalized


def torch_model_dtype(value: str | None) -> torch.dtype:
    normalized = normalize_model_dtype(value)
    return torch.float32 if normalized == "float32" else torch.float64
