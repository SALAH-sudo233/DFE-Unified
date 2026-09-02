import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
MASKFILL_PATH = ROOT / "models" / "maskfill.py"


def _method_node(name):
    tree = ast.parse(MASKFILL_PATH.read_text(encoding="utf-8"))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MaskFillModelVN"
    )
    return next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _self_method_calls(method, name):
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == name
    ]


class VectorOriginModelSourceContractTests(unittest.TestCase):
    def test_model_declares_nonpersistent_vector_origin_hook(self):
        source = MASKFILL_PATH.read_text(encoding="utf-8")
        self.assertIn("def set_science_vector_origin", source)
        self.assertIn("def _embed_compose", source)
        self.assertIn("_science_vector_origin", source)

    def test_inference_and_loss_use_shared_embedding_helper(self):
        for method_name in ("sample_focal", "get_loss"):
            with self.subTest(method=method_name):
                calls = _self_method_calls(
                    _method_node(method_name),
                    "_embed_compose",
                )
                self.assertEqual(len(calls), 1)

    def test_geometry_and_df_still_consume_original_compose_positions(self):
        sample_focal = _method_node("sample_focal")
        source = ast.unparse(sample_focal)
        self.assertIn("pos=compose_pos", source)
        self.assertIn(
            "self.compute_df_features_all(compose_pos, compose_feature, idx_protein)",
            source,
        )


try:
    from models.maskfill import MaskFillModelVN
except ImportError as import_error:
    MaskFillModelVN = None
    MODEL_IMPORT_ERROR = import_error
else:
    MODEL_IMPORT_ERROR = None


@unittest.skipIf(
    MaskFillModelVN is None,
    f"model dependencies unavailable: {MODEL_IMPORT_ERROR}",
)
class VectorOriginModelHookTests(unittest.TestCase):
    def setUp(self):
        self.model = MaskFillModelVN.__new__(MaskFillModelVN)
        torch.nn.Module.__init__(self.model)
        self.model.emb_dim = [4, 2]
        self.model.ligand_atom_emb = object()
        self.model.protein_atom_emb = object()
        object.__setattr__(self.model, "_science_vector_origin", "absolute")
        self.feature = torch.randn(3, 5)
        self.pos = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [4.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        self.idx_ligand = torch.tensor([0])
        self.idx_protein = torch.tensor([1, 2])

    def test_default_hook_passes_absolute_embedding_positions(self):
        with patch(
            "models.maskfill.embed_compose",
            return_value="embedded",
        ) as mocked:
            result = self.model._embed_compose(
                self.feature,
                self.pos,
                self.idx_ligand,
                self.idx_protein,
            )

        self.assertEqual(result, "embedded")
        self.assertIs(mocked.call_args.args[1], self.pos)

    def test_centered_hook_changes_only_embedding_positions(self):
        original = self.pos.clone()
        self.model.set_science_vector_origin("centered")

        with patch(
            "models.maskfill.embed_compose",
            return_value="embedded",
        ) as mocked:
            self.model._embed_compose(
                self.feature,
                self.pos,
                self.idx_ligand,
                self.idx_protein,
            )

        expected = self.pos - self.pos[self.idx_protein].mean(
            dim=0,
            keepdim=True,
        )
        self.assertTrue(torch.equal(mocked.call_args.args[1], expected))
        self.assertTrue(torch.equal(self.pos, original))

    def test_reset_restores_absolute_mode(self):
        self.model.set_science_vector_origin("zero")
        self.model.set_science_vector_origin()
        self.assertEqual(self.model._science_vector_origin, "absolute")


if __name__ == "__main__":
    unittest.main()
