"""Append-only, fsync-backed state ledger for every generation attempt."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping

from .contracts import canonical_json


TERMINAL_STATES = {"evaluated", "failed"}
ALLOWED_TRANSITIONS = {
    None: {"requested"},
    "requested": {"initialized", "failed"},
    "initialized": {"sampling", "failed"},
    "sampling": {"generated", "failed"},
    "generated": {"reconstructed", "failed"},
    "reconstructed": {"evaluated"},
    "evaluated": set(),
    "failed": set(),
}


class LedgerStateError(ValueError):
    pass


@dataclass(frozen=True)
class LedgerReplay:
    states: Mapping[str, str]
    records: tuple[dict[str, object], ...]
    truncated_final_line: bool


def _validate_transition(
    states: dict[str, str], record: Mapping[str, object]
) -> None:
    attempt_id = record.get("attempt_id")
    status = record.get("status")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise LedgerStateError("ledger record requires a non-empty attempt_id")
    if status not in ALLOWED_TRANSITIONS:
        raise LedgerStateError(f"unknown attempt status: {status}")
    prior = states.get(attempt_id)
    if prior in TERMINAL_STATES:
        raise LedgerStateError(f"attempt {attempt_id} is already terminal at {prior}")
    if status not in ALLOWED_TRANSITIONS[prior]:
        raise LedgerStateError(
            f"illegal transition for {attempt_id}: {prior!r} -> {status!r}"
        )


def replay_ledger(path: Path) -> LedgerReplay:
    path = Path(path)
    states: dict[str, str] = {}
    records: list[dict[str, object]] = []
    if not path.exists():
        return LedgerReplay(states, tuple(records), False)
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    truncated = bool(data and not data.endswith(b"\n"))
    for index, line in enumerate(lines):
        is_final = index == len(lines) - 1
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_final and truncated:
                break
            raise LedgerStateError(f"invalid ledger JSON at line {index + 1}") from exc
        if not isinstance(record, dict):
            raise LedgerStateError(f"ledger line {index + 1} must be an object")
        _validate_transition(states, record)
        states[str(record["attempt_id"])] = str(record["status"])
        records.append(record)
    return LedgerReplay(states, tuple(records), truncated)


class AttemptLedger:
    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        replay = replay_ledger(self.path) if resume else LedgerReplay({}, (), False)
        if resume and replay.truncated_final_line:
            raise LedgerStateError(
                "cannot append after a truncated final line; preserve evidence and repair explicitly"
            )
        self._states = dict(replay.states)
        mode = "ab" if resume else "xb"
        self._handle: IO[bytes] = self.path.open(mode)

    @property
    def states(self) -> Mapping[str, str]:
        return dict(self._states)

    def append(self, record: Mapping[str, object]) -> None:
        value = dict(record)
        if value.get("schema_version") != "phase0.v1":
            raise LedgerStateError("ledger schema_version must be phase0.v1")
        _validate_transition(self._states, value)
        self._handle.write(canonical_json(value))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._states[str(value["attempt_id"])] = str(value["status"])

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "AttemptLedger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
