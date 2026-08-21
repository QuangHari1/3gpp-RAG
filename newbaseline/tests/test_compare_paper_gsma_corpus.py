import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/compare_paper_gsma_corpus.py"
SPEC = importlib.util.spec_from_file_location("compare_paper_gsma_corpus", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparePaperGsmaCorpusTest(unittest.TestCase):
    def test_build_mapping_uses_specification_id_not_filename(self) -> None:
        paper_documents = [
            {
                "kind": "specification",
                "path": "Documents/22011-i50.docx",
            },
            {
                "kind": "specification",
                "path": "Documents/23700-05-i00.docx",
            },
            {
                "kind": "release_summary",
                "path": "Documents/rel_17.docx",
            },
        ]

        report = MODULE.build_mapping(
            paper_documents,
            {"22011": ["dataset/3gpp/marked/Rel-18/22_series/22011/raw.md"]},
        )

        self.assertEqual(report["paper_document_files"], 3)
        self.assertEqual(report["paper_unique_specification_ids"], 2)
        self.assertEqual(report["shared_specification_ids"], 1)
        self.assertEqual(report["gsma_raw_markdown_files"], 1)
        self.assertEqual(report["gsma_only_specification_ids"], 0)
        self.assertEqual(report["documents"][0]["match"], "shared_specification_id")
        self.assertEqual(report["documents"][1]["match"], "paper_only_specification_id")
        self.assertEqual(report["documents"][2]["match"], "not_a_specification")

    def test_collect_gsma_documents_uses_parent_directory_as_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_dir = Path(temporary_directory) / "Rel-18"
            raw_path = release_dir / "22_series" / "22011" / "raw.md"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text("text", encoding="utf-8")

            documents = MODULE.collect_gsma_documents(release_dir)

        self.assertEqual(len(documents), 1)
        self.assertTrue(documents["22011"][0].endswith("22_series/22011/raw.md"))

    def test_collect_gsma_documents_groups_version_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_dir = Path(temporary_directory) / "Rel-18"
            for variant in ("23700-05", "23700-08"):
                raw_path = release_dir / "23_series" / variant / "raw.md"
                raw_path.parent.mkdir(parents=True)
                raw_path.write_text("text", encoding="utf-8")

            documents = MODULE.collect_gsma_documents(release_dir)

        self.assertEqual(list(documents), ["23700"])
        self.assertEqual(len(documents["23700"]), 2)


if __name__ == "__main__":
    unittest.main()
