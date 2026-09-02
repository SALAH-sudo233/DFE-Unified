import unittest
from pathlib import Path

import torch

from dfe.science.model_precision import normalize_model_dtype, torch_model_dtype


class ModelPrecisionTests(unittest.TestCase):
    def test_default_and_explicit_float32(self):
        self.assertEqual(normalize_model_dtype(None), "float32")
        self.assertEqual(normalize_model_dtype("FLOAT32"), "float32")
        self.assertIs(torch_model_dtype("float32"), torch.float32)

    def test_float64_maps_to_torch_float64(self):
        self.assertEqual(normalize_model_dtype("float64"), "float64")
        self.assertIs(torch_model_dtype("float64"), torch.float64)

    def test_unknown_dtype_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model dtype"):
            normalize_model_dtype("bfloat16")

    def test_runtime_loads_checkpoint_before_dtype_conversion(self):
        source = (
            Path(__file__).parents[2] / "scripts" / "run_se3_audit.py"
        ).read_text(encoding="utf-8")
        load_index = source.index("model.load_state_dict(checkpoint[\"model\"], strict=True)")
        convert_index = source.index("model.to(device=device, dtype=torch_model_dtype(selected_dtype))")
        self.assertLess(load_index, convert_index)


if __name__ == "__main__":
    unittest.main()
