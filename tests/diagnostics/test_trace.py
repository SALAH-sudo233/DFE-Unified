import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from dfe.diagnostics.trace import TraceEvent, TraceWriter, summarize_tensor


IDENTITY = {
    "run_id": "run",
    "job_id": "main:10gs:20260901:D0",
    "attempt_id": "run:10gs:20260901:D0:0000",
    "pocket_id": "10gs",
    "seed": 20260901,
    "intervention": "D0",
}


class TraceTests(unittest.TestCase):
    def test_tensor_summary_is_finite_and_records_empty_shape(self):
        summary = summarize_tensor(torch.empty((0, 3)))
        self.assertEqual(summary["shape"], [0, 3])
        self.assertEqual(summary["count"], 0)
        for value in summary.values():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))

    def test_nonfinite_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_tensor(torch.tensor([float("nan")]))

    def test_event_requires_identity_and_payload(self):
        event = TraceEvent.new(
            **IDENTITY,
            step=0,
            event="frontier.indices",
            decision={"indices": []},
            monotonic_ns=10,
        )
        payload = event.to_dict()
        self.assertEqual(payload["schema_version"], "phase0.v1")
        self.assertEqual(payload["decision"], {"indices": []})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            TraceEvent.new(
                **IDENTITY,
                step=0,
                event="bad",
                monotonic_ns=10,
            )

    def test_writer_enforces_monotonic_time_and_encodes_empty_bonds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with TraceWriter(path) as writer:
                writer.append(
                    TraceEvent.new(
                        **IDENTITY,
                        step=1,
                        event="bond.indices",
                        decision={"indices": [], "types": []},
                        monotonic_ns=20,
                    )
                )
                with self.assertRaisesRegex(ValueError, "monotonic"):
                    writer.append(
                        TraceEvent.new(
                            **IDENTITY,
                            step=2,
                            event="termination",
                            decision={"reason": "done"},
                            monotonic_ns=19,
                        )
                    )
            rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
        self.assertEqual(rows[0]["decision"]["indices"], [])

    def test_resume_inherits_last_monotonic_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            event = TraceEvent.new(
                **IDENTITY,
                step=0,
                event="start",
                decision={"ok": True},
                monotonic_ns=100,
            )
            with TraceWriter(path) as writer:
                writer.append(event)
            with TraceWriter(path, resume=True) as writer:
                with self.assertRaisesRegex(ValueError, "monotonic"):
                    writer.append(
                        TraceEvent.new(
                            **IDENTITY,
                            step=1,
                            event="resume",
                            decision={"ok": True},
                            monotonic_ns=99,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
