"""Deterministic, restartable scheduling for explicit 0-4 device sets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence


ARM_ORDER = {
    "D0": 0,
    "D1": 1,
    "D2": 2,
    "D3": 3,
    "D4": 4,
    "D5-g0.25": 5,
    "D5-g0.5": 6,
    "D5-g1": 7,
    "D5-g1.5": 8,
}


@dataclass(frozen=True)
class JobStatus:
    state: str
    returncode: int | None = None

    def __post_init__(self) -> None:
        if self.state not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"unknown job state: {self.state}")


@dataclass(frozen=True)
class SchedulerResult:
    pending_count: int
    completed_count: int
    failed_count: int
    interrupted: bool


class Worker(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...


def parse_devices(value: str) -> tuple[str, ...]:
    value = value.strip()
    if value == "none":
        return ()
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or any(
        device != "cpu"
        and not (device.startswith("cuda:") and device[5:].isdigit())
        for device in devices
    ):
        raise ValueError("devices must be explicit: none, cpu, or cuda:N[,cuda:M]")
    if len(devices) != len(set(devices)):
        raise ValueError("duplicate devices are not allowed")
    if "cpu" in devices and len(devices) != 1:
        raise ValueError("cpu cannot be combined with CUDA devices")
    if len(devices) > 4:
        raise ValueError("Phase 0 accepts at most four explicit devices")
    return devices


def _job_sort_key(job: Mapping[str, object]) -> tuple[object, ...]:
    arm_id = str(job["arm_id"])
    return (
        str(job["pocket_id"]),
        int(job["seed"]),
        ARM_ORDER.get(arm_id, 999),
        arm_id,
        str(job["job_id"]),
    )


def build_pending_queue(
    jobs: Sequence[dict[str, object]],
    statuses: Mapping[str, JobStatus],
) -> list[dict[str, object]]:
    pending = [
        job
        for job in jobs
        if statuses.get(str(job["job_id"]), JobStatus("pending")).state == "pending"
    ]
    return sorted(pending, key=_job_sort_key)


def run_queue(
    queue: Sequence[dict[str, object]],
    devices: Sequence[str],
    launch: Callable[[dict[str, object], str], Worker],
    *,
    stop_requested: Callable[[], bool] | None = None,
    poll_interval: float = 0.0,
) -> SchedulerResult:
    pending = list(queue)
    initial_pending = len(pending)
    if not devices:
        return SchedulerResult(initial_pending, 0, 0, False)
    stop_requested = stop_requested or (lambda: False)
    available = list(devices)
    live: list[tuple[Worker, str, dict[str, object]]] = []
    completed = 0
    failed = 0
    interrupted = False
    while pending or live:
        if stop_requested():
            interrupted = True
        while pending and available and not interrupted:
            device = available.pop(0)
            job = pending.pop(0)
            live.append((launch(job, device), device, job))
        next_live = []
        for worker, device, job in live:
            returncode = worker.poll()
            if returncode is None:
                next_live.append((worker, device, job))
                continue
            available.append(device)
            if returncode == 0:
                completed += 1
            else:
                failed += 1
        live = next_live
        available.sort(key=lambda device: (device != "cpu", device))
        if live and poll_interval:
            time.sleep(poll_interval)
        if interrupted and not live:
            break
    return SchedulerResult(len(pending), completed, failed, interrupted)
