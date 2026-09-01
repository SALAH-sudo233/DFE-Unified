"""Stable, append-only autoregressive trace schemas and tensor summaries."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Mapping

import torch
from torch import Tensor

from .contracts import canonical_json


def summarize_tensor(value: Tensor) -> dict[str, object]:
    tensor = value.detach().cpu()
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError("tensor contains non-finite values")
    count = tensor.numel()
    summary: dict[str, object] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "count": count,
    }
    if count == 0:
        summary.update({"min": 0.0, "max": 0.0, "mean": 0.0, "l2": 0.0})
        return summary
    numeric = tensor.double()
    summary.update(
        {
            "min": float(numeric.min()),
            "max": float(numeric.max()),
            "mean": float(numeric.mean()),
            "l2": float(torch.linalg.vector_norm(numeric)),
        }
    )
    if not all(
        math.isfinite(value)
        for value in summary.values()
        if isinstance(value, float)
    ):
        raise ValueError("tensor summary contains non-finite values")
    return summary


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    job_id: str
    attempt_id: str
    pocket_id: str
    seed: int
    intervention: str
    step: int
    event: str
    monotonic_ns: int
    tensor: Mapping[str, object] | None = None
    decision: Mapping[str, object] | None = None
    schema_version: str = "phase0.v1"

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        job_id: str,
        attempt_id: str,
        pocket_id: str,
        seed: int,
        intervention: str,
        step: int,
        event: str,
        monotonic_ns: int,
        tensor: Mapping[str, object] | None = None,
        decision: Mapping[str, object] | None = None,
    ) -> "TraceEvent":
        if (tensor is None) == (decision is None):
            raise ValueError("trace event requires exactly one tensor or decision payload")
        identity = (run_id, job_id, attempt_id, pocket_id, intervention, event)
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError("trace identity fields must be non-empty strings")
        if step < 0 or monotonic_ns < 0:
            raise ValueError("trace step and monotonic time must be non-negative")
        return cls(
            run_id,
            job_id,
            attempt_id,
            pocket_id,
            seed,
            intervention,
            step,
            event,
            monotonic_ns,
            tensor,
            decision,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TraceWriter:
    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_monotonic_ns = -1
        if resume and self.path.exists():
            lines = self.path.read_text(encoding="ascii").splitlines()
            if lines:
                self._last_monotonic_ns = int(json.loads(lines[-1])["monotonic_ns"])
        self._handle: IO[bytes] = self.path.open("ab" if resume else "xb")

    def append(self, event: TraceEvent) -> None:
        if event.monotonic_ns <= self._last_monotonic_ns:
            raise ValueError("trace monotonic_ns must increase strictly")
        self._handle.write(canonical_json(event.to_dict()))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._last_monotonic_ns = event.monotonic_ns

    @property
    def last_monotonic_ns(self) -> int:
        return self._last_monotonic_ns

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
