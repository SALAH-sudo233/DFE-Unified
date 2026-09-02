"""Detached tensor snapshots for non-invasive model diagnostics."""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor


class TensorObserver:
    def __init__(self) -> None:
        self._events: dict[tuple[int, str], Tensor] = {}

    def observe(self, step: int, event: str, value: Tensor) -> None:
        key = (int(step), str(event))
        if key in self._events:
            raise ValueError(f"duplicate tensor event: {key}")
        self._events[key] = value.detach().clone().cpu()

    def get(self, step: int, event: str) -> Tensor:
        return self._events[(step, event)]

    def at_step(self, step: int):
        def callback(event: str, value: Tensor) -> None:
            self.observe(step, event, value)

        return callback

    def items(self) -> Iterator[tuple[tuple[int, str], Tensor]]:
        return iter(self._events.items())

    def keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(self._events)


def emit_tensor(
    observer: TensorObserver | None,
    step: int,
    event: str,
    value: Tensor,
) -> None:
    if observer is not None:
        observer.observe(step, event, value)
