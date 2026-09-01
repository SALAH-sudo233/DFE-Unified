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


if __name__ == "__main__":
    unittest.main()
