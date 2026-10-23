"""Small standard-library validation subset; does not replace the full pytest suite."""

import tempfile
import unittest
from pathlib import Path

from backend.ingestion.loader import extract, safe_filename
from backend.manifest import Manifest
from backend.rag.prompt import INSUFFICIENT, build_prompt, validate_citations


class OfflineChecks(unittest.TestCase):
    def test_text_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text("A reference passage", encoding="utf-8-sig")
            self.assertEqual(extract(path, ".md")[0].text, "A reference passage")
            self.assertIsNone(extract(path, ".md")[0].page_number)

    def test_empty_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("   ")
            with self.assertRaises(ValueError):
                extract(path, ".txt")

    def test_filenames(self):
        for filename in ["../../guide.pdf", "C:\\private\\guide.pdf"]:
            self.assertEqual(safe_filename(filename), "guide.pdf")
        with self.assertRaises(ValueError):
            safe_filename("bad.exe")

    def test_manifest_persistence_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.sqlite3"
            manifest = Manifest(path)
            record = {"document_id": "abc", "sha256": "hash", "status": "processing"}
            manifest.save(record)
            self.assertEqual(manifest.all(), [])
            self.assertEqual(Manifest(path).get("abc"), record)
            record["status"] = "ready"
            manifest.save(record)
            self.assertEqual(manifest.find_hash("hash"), record)
            self.assertEqual(len(manifest.all()), 1)
            manifest.delete("abc")
            self.assertEqual(Manifest(path).all(False), [])

    def test_citations(self):
        context, prompt, sources = build_prompt(
            "What?",
            [{"chunk_id": "abc:0", "filename": "guide.pdf", "page_number": 7, "text": "Reference"}],
        )
        self.assertIn("Reference", context)
        self.assertIn("What?", prompt)
        answer, sources, warnings = validate_citations("Evidence [S1], invented [S99]", sources)
        self.assertTrue(sources[0]["cited"])
        self.assertEqual(sources[0]["page_number"], 7)
        self.assertNotIn("[S99]", answer)
        self.assertTrue(warnings)

    def test_missing_citations(self):
        _, _, sources = build_prompt(
            "What?", [{"chunk_id": "abc:0", "filename": "guide.txt", "text": "Reference"}]
        )
        self.assertTrue(validate_citations("Uncited answer", sources)[2])
        self.assertFalse(validate_citations(INSUFFICIENT, sources)[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
