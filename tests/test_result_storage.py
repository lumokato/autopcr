import tempfile
import unittest
from pathlib import Path

from autopcr.util.result_storage import resolve_result_path


class ResultStorageTest(unittest.TestCase):
    def test_legacy_absolute_path_resolves_by_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            moved = result_dir / "moved.json"
            moved.write_text("{}", encoding="utf-8")

            resolved = resolve_result_path("/app/result/moved.json", result_dir)

            self.assertEqual(resolved, str(moved))

    def test_portable_result_path_uses_current_result_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)

            resolved = resolve_result_path("new.json", result_dir)

            self.assertEqual(resolved, str(result_dir / "new.json"))


if __name__ == "__main__":
    unittest.main()
