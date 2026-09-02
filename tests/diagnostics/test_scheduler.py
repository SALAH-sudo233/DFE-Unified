import unittest
from pathlib import Path

from dfe.diagnostics.scheduler import (
    JobStatus,
    build_pending_queue,
    parse_devices,
    run_queue,
)
from scripts.run_phase0_jobs import _command


def jobs(count=8):
    records = []
    arms = ("D0", "D1", "D2", "D3")
    for index in range(count):
        arm = arms[index % len(arms)]
        records.append(
            {
                "job_id": f"smoke:p{index // 4:02d}:20260901:{arm}",
                "stage": "smoke",
                "pocket_id": f"p{index // 4:02d}",
                "seed": 20260901,
                "arm_id": arm,
            }
        )
    return records


class FakeWorker:
    def __init__(self, job, device, tracker):
        self.job = job
        self.device = device
        self.tracker = tracker
        self.poll_count = 0
        self.returncode = None
        tracker["launched"].append((job["job_id"], device))
        tracker["live"] += 1
        tracker["max_live"] = max(tracker["max_live"], tracker["live"])

    def poll(self):
        self.poll_count += 1
        if self.poll_count >= 2 and self.returncode is None:
            self.returncode = 0
            self.tracker["live"] -= 1
        return self.returncode


class SchedulerTests(unittest.TestCase):
    def test_parse_devices_requires_explicit_cpu_or_cuda_ids(self):
        self.assertEqual(parse_devices("cpu"), ("cpu",))
        self.assertEqual(parse_devices("cuda:0,cuda:2"), ("cuda:0", "cuda:2"))
        with self.assertRaisesRegex(ValueError, "explicit"):
            parse_devices("cuda")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_devices("cuda:0,cuda:0")

    def test_zero_devices_lists_pending_without_launching(self):
        queue = build_pending_queue(jobs(2), {})
        tracker = {"launched": [], "live": 0, "max_live": 0}
        result = run_queue(queue, (), lambda job, device: FakeWorker(job, device, tracker))
        self.assertEqual(result.pending_count, 2)
        self.assertEqual(tracker["launched"], [])

    def test_one_gpu_runs_serially_and_four_never_exceed_four(self):
        for devices, expected in (("cuda:0", 1), ("cuda:0,cuda:1,cuda:2,cuda:3", 4)):
            with self.subTest(devices=devices):
                tracker = {"launched": [], "live": 0, "max_live": 0}
                result = run_queue(
                    build_pending_queue(jobs(8), {}),
                    parse_devices(devices),
                    lambda job, device: FakeWorker(job, device, tracker),
                )
                self.assertEqual(result.completed_count, 8)
                self.assertLessEqual(tracker["max_live"], expected)

    def test_completed_and_failed_jobs_are_not_dispatched(self):
        records = jobs(4)
        states = {
            records[0]["job_id"]: JobStatus("completed"),
            records[1]["job_id"]: JobStatus("failed"),
        }
        queue = build_pending_queue(records, states)
        self.assertEqual([job["job_id"] for job in queue], [records[2]["job_id"], records[3]["job_id"]])

    def test_d0_d1_paired_order_is_stable(self):
        shuffled = list(reversed(jobs(8)))
        queue = build_pending_queue(shuffled, {})
        ids = [job["job_id"] for job in queue]
        self.assertEqual(ids[:2], ["smoke:p00:20260901:D0", "smoke:p00:20260901:D1"])
        self.assertEqual(ids[4:6], ["smoke:p01:20260901:D0", "smoke:p01:20260901:D1"])

    def test_stop_request_prevents_new_dispatch(self):
        tracker = {"launched": [], "live": 0, "max_live": 0}
        polls = {"count": 0}

        def stop_requested():
            polls["count"] += 1
            return polls["count"] > 1

        result = run_queue(
            build_pending_queue(jobs(8), {}),
            parse_devices("cuda:0,cuda:1"),
            lambda job, device: FakeWorker(job, device, tracker),
            stop_requested=stop_requested,
        )
        self.assertTrue(result.interrupted)
        self.assertLess(len(tracker["launched"]), 8)

    def test_cuda_worker_is_remapped_to_process_local_zero(self):
        job = jobs(1)[0]
        command, visible = _command(Path("manifest.json"), job, "cuda:2", False)
        self.assertEqual(visible, "2")
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")


if __name__ == "__main__":
    unittest.main()
