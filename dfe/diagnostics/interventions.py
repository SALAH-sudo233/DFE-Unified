"""Typed, inference-only interventions for the frozen DF 500K model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch import Tensor, nn


FROZEN_D5_GATES = (0.25, 0.5, 1.0, 1.5)


class DirectionFieldModule(Protocol):
    def __call__(self, *inputs: Tensor) -> Tensor: ...

    def raw_features(self, *inputs: Tensor) -> Tensor: ...

    def project_features(self, raw_features: Tensor) -> Tensor: ...


TensorObserver = Callable[[str, Tensor], None]


@dataclass(frozen=True)
class DFIntervention:
    name: str = "D0"
    gate: float = 1.0
    zero_direction: bool = False
    shuffle_seed: int | None = None
    alternate_raw: Tensor | None = None

    @classmethod
    def from_arm(
        cls,
        name: str,
        *,
        gate: float | None = None,
        shuffle_seed: int | None = None,
        alternate_raw: Tensor | None = None,
    ) -> "DFIntervention":
        if name not in {"D0", "D1", "D2", "D3", "D4", "D5"}:
            raise ValueError(f"unknown DF intervention: {name}")
        if name == "D0":
            return cls(name=name)
        if name == "D1":
            return cls(name=name, gate=0.0)
        if name == "D2":
            return cls(name=name, zero_direction=True)
        if name == "D3":
            if shuffle_seed is None:
                raise ValueError("D3 requires a deterministic shuffle seed")
            return cls(name=name, shuffle_seed=shuffle_seed)
        if name == "D4":
            if alternate_raw is None:
                raise ValueError("D4 requires alternate raw features")
            return cls(name=name, alternate_raw=alternate_raw)
        if gate not in FROZEN_D5_GATES:
            raise ValueError(f"D5 gate must be one of {FROZEN_D5_GATES}")
        return cls(name=name, gate=float(gate))

    def apply_raw(self, raw: Tensor) -> Tensor:
        value = raw.clone()
        if self.zero_direction:
            value[..., 1:4] = 0
        if self.shuffle_seed is not None:
            generator = torch.Generator(device=value.device)
            generator.manual_seed(self.shuffle_seed)
            permutation = torch.randperm(
                value.shape[0], generator=generator, device=value.device
            )
            value = value[permutation]
        if self.alternate_raw is not None:
            alternate = self.alternate_raw.to(value)
            if alternate.shape != value.shape:
                raise ValueError(
                    f"D4 alternate raw shape {alternate.shape} does not match {value.shape}"
                )
            value = alternate.clone()
        return value

    def apply_projected(self, value: Tensor) -> Tensor:
        return value * self.gate


def _observe(observer: TensorObserver | None, name: str, value: Tensor) -> None:
    if observer is not None:
        observer(name, value.detach().clone().cpu())


def compute_df_with_diagnostics(
    df_module: DirectionFieldModule,
    final_projection: nn.Module,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    intervention: DFIntervention | None = None,
    observer: TensorObserver | None = None,
) -> Tensor:
    if intervention is None and observer is None:
        return final_projection(df_module(*inputs))

    effective = intervention or DFIntervention.from_arm("D0")
    raw = df_module.raw_features(*inputs)
    _observe(observer, "df.raw", raw)
    hidden = df_module.project_features(effective.apply_raw(raw))
    _observe(observer, "df.hidden", hidden)
    projected = effective.apply_projected(final_projection(hidden))
    _observe(observer, "df.projected", projected)
    return projected
