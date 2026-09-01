import unittest

import torch

from dfe.diagnostics.observer import TensorObserver, emit_tensor


class TensorObserverTests(unittest.TestCase):
    def test_unset_observer_returns_without_allocating_a_copy(self):
        value = torch.tensor([1.0], requires_grad=True)
        self.assertIsNone(emit_tensor(None, 0, "encoder.scalar", value))

    def test_observer_detaches_clones_to_cpu(self):
        value = torch.tensor([1.0, 2.0], requires_grad=True)
        observer = TensorObserver()
        observer.observe(3, "encoder.scalar", value)
        snapshot = observer.get(3, "encoder.scalar")
        self.assertEqual(snapshot.device.type, "cpu")
        self.assertFalse(snapshot.requires_grad)
        value.detach().zero_()
        torch.testing.assert_close(snapshot, torch.tensor([1.0, 2.0]))

    def test_duplicate_step_event_key_is_rejected(self):
        observer = TensorObserver()
        observer.observe(0, "df.raw", torch.ones(1))
        with self.assertRaisesRegex(ValueError, "duplicate tensor event"):
            observer.observe(0, "df.raw", torch.zeros(1))

    def test_mutating_observed_tensor_does_not_mutate_model_output(self):
        output = torch.tensor([[1.0, 2.0]])
        observer = TensorObserver()
        observer.observe(0, "element.logits", output)
        observer.get(0, "element.logits").zero_()
        torch.testing.assert_close(output, torch.tensor([[1.0, 2.0]]))

    def test_step_bound_callable_supports_existing_df_callback(self):
        observer = TensorObserver()
        callback = observer.at_step(8)
        callback("df.raw", torch.ones(2, 8))
        self.assertEqual(observer.get(8, "df.raw").shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
