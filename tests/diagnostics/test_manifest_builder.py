import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from dfe.diagnostics.io import (
    InputManifestError,
    build_phase0_manifest,
    git_state,
    resolve_pdbbind_inputs,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "diagnostics" / "phase0_df500k.yaml"


def write_pocket(root: Path, pocket_id: str) -> None:
    directory = root / pocket_id
    directory.mkdir(parents=True)
    (directory / f"{pocket_id}_protein.pdb").write_text("ATOM protein\n")
    (directory / f"{pocket_id}_pocket.pdb").write_text("ATOM pocket\n")
    (directory / f"{pocket_id}_ligand.sdf").write_text("ligand\n$$$$\n")


class InputResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_pocket(self.root, "10gs")

    def tearDown(self):
        self.temporary.cleanup()

    def test_resolved_inputs_have_logical_names_sizes_and_hashes(self):
        records = resolve_pdbbind_inputs(self.root, "10gs")
        self.assertEqual({record.role for record in records}, {"protein", "pocket", "ligand"})
        for record in records:
            self.assertFalse(Path(record.logical_path).is_absolute())
            self.assertEqual(record.size, record.source_path.stat().st_size)
            self.assertEqual(record.sha256, sha256_file(record.source_path))

    def test_missing_required_input_is_rejected_by_role(self):
        for role in ("protein", "pocket", "ligand"):
            with self.subTest(role=role):
                path = self.root / "10gs" / f"10gs_{role}.{'sdf' if role == 'ligand' else 'pdb'}"
                content = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(InputManifestError, role):
                    resolve_pdbbind_inputs(self.root, "10gs")
                path.write_bytes(content)


class ManifestBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "pdbbind"
        write_pocket(self.dataset, "10gs")
        self.centers = self.root / "centers.json"
        self.centers.write_text(json.dumps({"10gs": [1.0, 2.0, 3.0]}))
        self.repository = self.root / "source"
        self.repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "phase0@example.invalid"],
            cwd=self.repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Phase0 Test"],
            cwd=self.repository,
            check=True,
        )
        (self.repository / "tracked.txt").write_text("baseline\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "baseline"],
            cwd=self.repository,
            check=True,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_dirty_git_state_is_recorded_and_require_clean_rejects_it(self):
        (self.repository / "untracked.txt").write_text("dirty\n")
        state = git_state(self.repository)
        self.assertTrue(state.is_dirty)
        self.assertTrue(any(line.endswith("untracked.txt") for line in state.porcelain))

        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(InputManifestError, "clean Git tree"):
            build_phase0_manifest(
                CONFIG,
                self.dataset,
                self.centers,
                output,
                repository_root=self.repository,
                require_clean=True,
                expected_pocket_count=1,
            )
        self.assertFalse(output.exists())

    def test_builder_creates_hash_bound_outputs_once(self):
        output = self.root / "run"
        manifest = build_phase0_manifest(
            CONFIG,
            self.dataset,
            self.centers,
            output,
            repository_root=self.repository,
            expected_pocket_count=1,
        )
        self.assertEqual(manifest["pocket_count"], 1)
        self.assertEqual(manifest["smoke_candidate_job_count"], 9)
        self.assertEqual(manifest["main_candidate_job_count"], 27)
        self.assertFalse(manifest["git"]["is_dirty"])
        self.assertTrue((output / "pockets.jsonl").is_file())
        self.assertTrue((output / "jobs.jsonl").is_file())
        self.assertIn("sampling_policy", manifest["inputs"])
        self.assertEqual(
            sha256_file(output / "pockets.jsonl"),
            manifest["artifacts"]["pockets"]["sha256"],
        )
        with self.assertRaises(FileExistsError):
            build_phase0_manifest(
                CONFIG,
                self.dataset,
                self.centers,
                output,
                repository_root=self.repository,
                expected_pocket_count=1,
            )


if __name__ == "__main__":
    unittest.main()
