import tempfile
import unittest
from pathlib import Path

import numpy as np

from dfe.diagnostics.openness import (
    compute_openness,
    fibonacci_directions,
    parse_pdb_heavy_atoms,
    ray_sphere_blocked,
    select_smoke_records,
)


def random_rotation(seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


class OpennessGeometryTests(unittest.TestCase):
    def setUp(self):
        directions = fibonacci_directions(256)
        self.coords = directions * 4.0
        self.elements = ["C"] * len(self.coords)
        self.center = np.array([0.3, -0.2, 0.1])

    def test_fibonacci_directions_are_deterministic_unit_vectors(self):
        first = fibonacci_directions(2048)
        second = fibonacci_directions(2048)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-15)

    def test_openness_is_stable_under_rotation_and_translation(self):
        result = compute_openness(self.coords, self.elements, self.center)
        rotation = random_rotation()
        shift = np.array([3.0, -5.0, 2.0])
        moved = compute_openness(
            self.coords @ rotation.T + shift,
            self.elements,
            self.center @ rotation.T + shift,
        )
        self.assertAlmostEqual(result.nearest_distance, moved.nearest_distance, places=12)
        self.assertLessEqual(abs(result.openness - moved.openness), 0.01)

    def test_empty_environment_is_fully_open(self):
        result = compute_openness(np.empty((0, 3)), [], np.zeros(3))
        self.assertEqual(result.blocked_rays, 0)
        self.assertEqual(result.enclosure, 0.0)
        self.assertEqual(result.openness, 1.0)
        self.assertIsNone(result.nearest_distance)

    def test_enclosing_shell_blocks_more_than_half_shell(self):
        directions = fibonacci_directions(1024)
        shell = compute_openness(directions * 3.0, ["C"] * 1024, np.zeros(3))
        half = compute_openness(
            directions[directions[:, 2] >= 0] * 3.0,
            ["C"] * np.count_nonzero(directions[:, 2] >= 0),
            np.zeros(3),
        )
        self.assertGreater(shell.enclosure, 0.95)
        self.assertGreater(half.openness, shell.openness)

    def test_ray_intersections_respect_cutoff_and_radius(self):
        directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        blocked = ray_sphere_blocked(
            directions,
            np.array([[5.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            np.array([1.0, 2.0]),
            cutoff=12.0,
        )
        np.testing.assert_array_equal(blocked, [True, False])


class OpennessInputTests(unittest.TestCase):
    def test_pdb_parser_excludes_hydrogen_and_counts_unknown_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pocket.pdb"
            path.write_text(
                "ATOM      1  C   GLY A   1       1.000   2.000   3.000  1.00 20.00           C  \n"
                "ATOM      2  H   GLY A   1       2.000   2.000   3.000  1.00 20.00           H  \n"
                "HETATM    3  XX  UNK A   2       3.000   2.000   3.000  1.00 20.00           Xx \n",
                encoding="ascii",
            )
            parsed = parse_pdb_heavy_atoms(path)
        self.assertEqual(parsed.coordinates.shape, (2, 3))
        self.assertEqual(parsed.elements, ("C", "X"))
        self.assertEqual(parsed.unknown_element_count, 1)

    def test_smoke_selection_and_wrong_pocket_pairing_are_deterministic(self):
        records = [
            {"pocket_id": f"p{i:02d}", "openness": float(i)} for i in range(30)
        ]
        smoke, enriched = select_smoke_records(records)
        self.assertEqual(
            [record["pocket_id"] for record in smoke],
            ["p00", "p01", "p14", "p15", "p28", "p29"],
        )
        by_id = {record["pocket_id"]: record for record in enriched}
        self.assertEqual(by_id["p00"]["wrong_pocket_id"], "p01")
        self.assertEqual(by_id["p09"]["wrong_pocket_id"], "p00")
        self.assertEqual(by_id["p10"]["openness_tertile"], "middle")
        self.assertEqual(by_id["p20"]["openness_tertile"], "open")


if __name__ == "__main__":
    unittest.main()
