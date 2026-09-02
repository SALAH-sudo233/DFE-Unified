import pathlib
import tempfile
import unittest

from scripts.verify_repository import scan_forbidden_content, scan_forbidden_names


class RepositorySecurityScanTests(unittest.TestCase):
    def test_detects_high_confidence_secret_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "config.py"
            key = "pass" + "word"
            candidate.write_text(
                f'{key} = "not-a-real-secret"\n', encoding="utf-8"
            )

            violations = scan_forbidden_content(root, [candidate])

            self.assertEqual(len(violations), 1)
            self.assertIn("secret assignment", violations[0])

    def test_detects_forbidden_operational_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "ssh_upload_results.py"
            candidate.write_text("print('example')\n", encoding="utf-8")

            violations = scan_forbidden_names(root, [candidate])

            self.assertEqual(violations, ["forbidden filename: ssh_upload_results.py"])


class ScienceSourceVerificationTests(unittest.TestCase):
    def test_vector_origin_authorities_and_tests_are_required(self):
        source = (
            pathlib.Path(__file__).parents[1]
            / "scripts"
            / "verify_repository.py"
        ).read_text(encoding="utf-8")
        required = (
            "dfe/science/vector_origin.py",
            "tests/science/test_vector_origin.py",
            "tests/science/test_vector_origin_model_hook.py",
            "docs/superpowers/specs/2026-09-02-se3-vector-origin-candidates-design.md",
            "docs/superpowers/plans/2026-09-02-se3-vector-origin-candidates.md",
            "dfe/science/model_precision.py",
            "tests/science/test_model_precision.py",
            "docs/superpowers/specs/2026-09-02-zero-fp64-science-design.md",
            "docs/superpowers/plans/2026-09-02-zero-fp64-science.md",
        )

        for path in required:
            with self.subTest(path=path):
                self.assertIn(path, source)


if __name__ == "__main__":
    unittest.main()
