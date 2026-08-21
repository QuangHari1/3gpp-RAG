import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"


def load_script(name: str):
    script_path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("prepare_embedding_selection.py")
EMBED = load_script("embed_chunks.py")


class EmbeddingSelectionTest(unittest.TestCase):
    def test_paper_variant_matching_and_single_candidate_fallback(self) -> None:
        exact_row = {
            "paper_filename": "23700-05-i00.docx",
            "gsma_raw_markdown_paths": [
                "dataset/3gpp/marked/Rel-18/23_series/23700-05/raw.md",
                "dataset/3gpp/marked/Rel-18/23_series/23700-08/raw.md",
            ],
        }
        fallback_row = {
            "paper_filename": "22261-ic0.docx",
            "gsma_raw_markdown_paths": [
                "dataset/3gpp/marked/Rel-18/22_series/22261/raw.md",
            ],
        }

        self.assertEqual(
            PREPARE.exact_variant_paths(exact_row),
            ["dataset/3gpp/marked/Rel-18/23_series/23700-05/raw.md"],
        )
        self.assertEqual(PREPARE.exact_variant_paths(fallback_row), [])

    def test_embed_loader_filters_by_document_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chunk_path = Path(temporary_directory) / "ChunkSeries23.json"
            chunk_path.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {"text": "keep", "document_key": "23_series/23700-05"},
                            {"text": "drop", "document_key": "23_series/23700-08"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            chunks = EMBED.load_chunks(chunk_path, {"23_series/23700-05"})
            metadata_path = Path(temporary_directory) / "EmbeddingsSeries23.metadata.jsonl"
            EMBED.write_row_metadata(metadata_path, chunks)
            metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([chunk["text"] for chunk in chunks], ["keep"])
        self.assertEqual(chunks[0]["_source_chunk_index"], 0)
        self.assertEqual(metadata_rows[0]["embedding_row"], 0)
        self.assertEqual(metadata_rows[0]["source_chunk_index"], 0)
        self.assertEqual(metadata_rows[0]["document_key"], "23_series/23700-05")


if __name__ == "__main__":
    unittest.main()
