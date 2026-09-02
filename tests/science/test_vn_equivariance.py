import unittest
from pathlib import Path

try:
    import torch
    from models.invariant import MessageModule, VNLinear
except ModuleNotFoundError as exc:  # optional CUDA/PyG stack is absent on local CPU runners
    torch = None
    MessageModule = VNLinear = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class VNSourceContractTests(unittest.TestCase):
    def test_message_edge_vector_projection_has_no_coordinate_bias(self):
        source = (Path(__file__).parents[2] / "models" / "invariant.py").read_text(encoding="utf-8")
        self.assertIn(
            "self.edge_vnlinear = VNLinear(hid_vec, out_vec, bias=True, apply_bias=False)",
            source,
        )

    def test_atom_embedding_does_not_apply_coordinate_bias(self):
        source = (Path(__file__).parents[2] / "models" / "embedding.py").read_text(encoding="utf-8")
        self.assertIn("F.linear(vec_emb, self.emb_vec.weight, None)", source)

    def test_encoder_uses_equivariant_vector_normalization(self):
        source = (Path(__file__).parents[2] / "models" / "encoders" / "cftfm.py").read_text(encoding="utf-8")
        self.assertIn("self.layernorm_vec = EquivariantVectorNorm(hidden_channels[1])", source)


@unittest.skipIf(torch is None, f"VN tests require the model dependency stack: {_IMPORT_ERROR}")
class VNEquivarianceTests(unittest.TestCase):

    def test_bias_free_vnlinear_commutes_with_rotation(self):
        module = VNLinear(4, 7, bias=False).double()
        vector = torch.randn(5, 4, 3, dtype=torch.float64)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        actual = module(vector @ rotation.T)
        expected = module(vector) @ rotation.T
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
