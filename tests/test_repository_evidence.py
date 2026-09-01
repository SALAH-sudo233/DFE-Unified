import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RepositoryEvidenceTests(unittest.TestCase):
    def test_code_manifest_matches_files(self):
        manifest_path = ROOT / "evidence" / "code-manifest.json"
        self.assertTrue(manifest_path.is_file(), "code manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["upstream_commit"],
            "836a0c4ce487297ad24bc54ac2ebd163de13242c",
        )
        self.assertGreaterEqual(len(manifest["files"]), 18)
        for record in manifest["files"]:
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(path.stat().st_size, record["size"])
            self.assertEqual(sha256(path), record["sha256"])

    def test_checkpoint_manifest_matches_500k_artifact(self):
        manifest_path = ROOT / "artifacts" / "MANIFEST.json"
        self.assertTrue(manifest_path.is_file(), "artifact manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = ROOT / manifest["checkpoint"]["path"]
        self.assertEqual(manifest["checkpoint"]["iteration"], 500000)
        self.assertEqual(
            manifest["checkpoint"]["sha256"],
            "34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0",
        )
        self.assertEqual(sha256(checkpoint), manifest["checkpoint"]["sha256"])

    def test_partial_run_contains_exact_observed_scope(self):
        result_root = ROOT / "results" / "df-500k-21-pocket"
        provenance_path = result_root / "provenance.json"
        self.assertTrue(provenance_path.is_file(), "result provenance is missing")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
        raw_paths = sorted((result_root / "per-pocket").glob("*/docking_results.json"))
        raw_results = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
        self.assertEqual(provenance["status"], "partial")
        self.assertEqual(provenance["requested_pockets"], 30)
        self.assertEqual(provenance["completed_pockets"], 21)
        self.assertEqual(provenance["evaluated_records"], 2331)
        self.assertEqual(len(summary), 21)
        self.assertEqual(len(raw_paths), 21)
        self.assertEqual(sum(map(len, raw_results)), 2331)
        self.assertEqual(sum(item["total_molecules"] for item in summary.values()), 2331)

    def test_forbidden_operational_files_are_absent(self):
        forbidden_fragments = (
            "ssh_",
            "paramiko",
            "heartbeat",
            "credentials",
            "private_key",
        )
        tracked_candidates = [path for path in ROOT.rglob("*") if path.is_file()]
        relative_names = [str(path.relative_to(ROOT)).lower() for path in tracked_candidates]
        violations = [
            name for name in relative_names
            if ".git/" not in name and any(fragment in name for fragment in forbidden_fragments)
        ]
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
