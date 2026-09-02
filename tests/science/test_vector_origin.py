import unittest

import torch

from dfe.science.vector_origin import (
    normalize_vector_origin_mode,
    vector_embedding_positions,
)


class VectorOriginTests(unittest.TestCase):
    def setUp(self):
        self.pos = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [4.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        self.idx_protein = torch.tensor([1, 2])

    def test_none_and_absolute_preserve_the_existing_tensor(self):
        for mode in (None, "absolute"):
            with self.subTest(mode=mode):
                actual = vector_embedding_positions(
                    self.pos,
                    self.idx_protein,
                    mode,
                )
                self.assertTrue(torch.equal(actual, self.pos))
                self.assertEqual(actual.data_ptr(), self.pos.data_ptr())

    def test_centered_uses_protein_centroid_for_all_atoms(self):
        original = self.pos.clone()
        expected = self.pos - self.pos[self.idx_protein].mean(
            dim=0,
            keepdim=True,
        )

        actual = vector_embedding_positions(
            self.pos,
            self.idx_protein,
            "centered",
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(self.pos, original))

    def test_centered_obeys_rigid_transform_law(self):
        rotation = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = torch.tensor([7.0, -3.0, 2.0])
        moved = self.pos @ rotation.T + translation
        reference = vector_embedding_positions(
            self.pos,
            self.idx_protein,
            "centered",
        )

        actual = vector_embedding_positions(
            moved,
            self.idx_protein,
            "centered",
        )

        self.assertTrue(
            torch.allclose(
                actual,
                reference @ rotation.T,
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_zero_returns_independent_zeros(self):
        actual = vector_embedding_positions(
            self.pos,
            self.idx_protein,
            "zero",
        )

        self.assertTrue(torch.equal(actual, torch.zeros_like(self.pos)))
        self.assertNotEqual(actual.data_ptr(), self.pos.data_ptr())

    def test_centered_rejects_empty_protein_selection(self):
        with self.assertRaisesRegex(ValueError, "protein"):
            vector_embedding_positions(
                self.pos,
                torch.empty(0, dtype=torch.long),
                "centered",
            )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vector-origin"):
            normalize_vector_origin_mode("learned")

    def test_invalid_position_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            vector_embedding_positions(
                torch.zeros(3),
                self.idx_protein,
                "absolute",
            )


if __name__ == "__main__":
    unittest.main()
