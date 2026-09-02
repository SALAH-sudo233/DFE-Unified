#!/usr/bin/env python3
"""Schedule manifest-declared Phase 0 jobs on 0-4 explicit devices."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import canonical_json  # noqa: E402
from dfe.diagnostics.ledger import replay_ledger  # noqa: E402
from dfe.diagnostics.scheduler import (  # noqa: E402
    JobStatus,
    build_pending_queue,
    parse_devices,
    run_queue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "main"), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line]


def _selected_jobs(run_root: Path, manifest: dict[str, object], stage: str):
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    jobs = [job for job in _read_jsonl(jobs_path) if job["stage"] == stage]
    if stage == "smoke":
        smoke_path = run_root / "smoke-pockets.json"
        if not smoke_path.is_file():
            raise ValueError("smoke requires frozen smoke-pockets.json")
        pocket_ids = set(json.loads(smoke_path.read_text(encoding="ascii"))["pocket_ids"])
        jobs = [job for job in jobs if job["pocket_id"] in pocket_ids]
    else:
        gate_path = run_root / "gate-smoke.json"
        if not gate_path.is_file():
            raise ValueError("main requires gate-smoke.json")
        gate = json.loads(gate_path.read_text(encoding="ascii"))
        if gate.get("status") != "pass":
            raise ValueError("main requires a passing smoke gate")
        retained = set(gate["retained_arm_ids"])
        jobs = [job for job in jobs if job["arm_id"] in retained]
    return jobs


def _job_status(run_root: Path, job: dict[str, object]) -> JobStatus:
    path = run_root / "jobs" / str(job["job_id"]) / "attempts.jsonl"
    if not path.is_file():
        return JobStatus("pending")
    replay = replay_ledger(path)
    if replay.truncated_final_line:
        return JobStatus("failed")
    states = tuple(replay.states.values())
    if len(states) != int(job["attempt_count"]):
        return JobStatus("pending")
    if all(state in {"evaluated", "failed"} for state in states):
        return JobStatus("completed")
    return JobStatus("pending")


def _command(manifest: Path, job: dict[str, object], device: str, resume: bool):
    visible_device = None
    worker_device = device
    if device.startswith("cuda:"):
        visible_device = device.split(":", 1)[1]
        worker_device = "cuda:0"
    command = [
        sys.executable,
        str(ROOT / "sample_diagnostic.py"),
        "--manifest",
        str(manifest),
        "--job-id",
        str(job["job_id"]),
        "--device",
        worker_device,
    ]
    if resume:
        command.append("--resume")
    return command, visible_device


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    jobs = _selected_jobs(run_root, manifest, args.stage)
    statuses = {str(job["job_id"]): _job_status(run_root, job) for job in jobs}
    queue = build_pending_queue(jobs, statuses)
    devices = parse_devices(args.devices)
    commands = []
    for index, job in enumerate(queue):
        device = devices[index % len(devices)] if devices else "unassigned"
        resume = (run_root / "jobs" / str(job["job_id"]) / "attempts.jsonl").is_file()
        command, visible = _command(manifest_path, job, device, resume)
        commands.append(
            {
                "job_id": job["job_id"],
                "device": device,
                "cuda_visible_devices": visible,
                "command": command,
            }
        )
    if args.dry_run or not devices:
        print(json.dumps({"pending_count": len(queue), "commands": commands}, indent=2))
        return 0

    stopped = {"value": False}

    def handle_signal(signum, frame):
        del signum, frame
        stopped["value"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    events_path = run_root / "scheduler-events.jsonl"

    def append_event(value: dict[str, object]) -> None:
        with events_path.open("ab") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())

    def launch(job, device):
        resume = (run_root / "jobs" / str(job["job_id"]) / "attempts.jsonl").is_file()
        command, visible = _command(manifest_path, job, device, resume)
        environment = os.environ.copy()
        if visible is not None:
            environment["CUDA_VISIBLE_DEVICES"] = visible
        append_event(
            {
                "schema_version": "phase0.v1",
                "event": "launched",
                "job_id": job["job_id"],
                "device": device,
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        return subprocess.Popen(command, cwd=ROOT, env=environment)

    result = run_queue(
        queue,
        devices,
        launch,
        stop_requested=lambda: stopped["value"],
        poll_interval=0.25,
    )
    append_event(
        {
            "schema_version": "phase0.v1",
            "event": "scheduler_stopped",
            "pending_count": result.pending_count,
            "completed_count": result.completed_count,
            "failed_count": result.failed_count,
            "interrupted": result.interrupted,
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    if result.interrupted:
        return 130
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
