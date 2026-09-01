import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dfe.diagnostics.ledger import (
    AttemptLedger,
    LedgerStateError,
    replay_ledger,
)


def transition(attempt_id: str, status: str) -> dict[str, object]:
    return {
        "schema_version": "phase0.v1",
        "attempt_id": attempt_id,
        "status": status,
    }


class AttemptLedgerTests(unittest.TestCase):
    def test_valid_lifecycle_replays_to_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            with AttemptLedger(path) as ledger:
                for status in (
                    "requested",
                    "initialized",
                    "sampling",
                    "generated",
                    "reconstructed",
                    "evaluated",
                ):
                    ledger.append(transition("a0", status))
            replay = replay_ledger(path)
        self.assertEqual(replay.states, {"a0": "evaluated"})
        self.assertFalse(replay.truncated_final_line)

    def test_illegal_backwards_transition_and_duplicate_terminal_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            with AttemptLedger(path) as ledger:
                ledger.append(transition("a0", "requested"))
                ledger.append(transition("a0", "initialized"))
                ledger.append(transition("a0", "sampling"))
                with self.assertRaisesRegex(LedgerStateError, "illegal transition"):
                    ledger.append(transition("a0", "initialized"))
                ledger.append(transition("a0", "failed"))
                with self.assertRaisesRegex(LedgerStateError, "terminal"):
                    ledger.append(transition("a0", "failed"))

    def test_every_append_flushes_and_fsyncs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            with mock.patch("dfe.diagnostics.ledger.os.fsync") as fsync:
                with AttemptLedger(path) as ledger:
                    ledger.append(transition("a0", "requested"))
                    self.assertEqual(fsync.call_count, 1)
                    self.assertEqual(path.read_text(encoding="ascii").count("\n"), 1)

    def test_replay_ignores_only_a_truncated_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            path.write_text(
                json.dumps(transition("a0", "requested")) + "\n" + '{"attempt_id":"a0"',
                encoding="ascii",
            )
            replay = replay_ledger(path)
        self.assertEqual(replay.states, {"a0": "requested"})
        self.assertTrue(replay.truncated_final_line)

    def test_failed_is_allowed_from_every_nonterminal_sampling_stage(self):
        for prior in ("requested", "initialized", "sampling", "generated"):
            with self.subTest(prior=prior), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "attempts.jsonl"
                with AttemptLedger(path) as ledger:
                    ledger.append(transition("a0", "requested"))
                    for status in ("initialized", "sampling", "generated"):
                        if status == prior:
                            break
                        ledger.append(transition("a0", status))
                    if prior != "requested":
                        ledger.append(transition("a0", prior))
                    ledger.append(transition("a0", "failed"))


if __name__ == "__main__":
    unittest.main()
