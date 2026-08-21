"""Focused offline checks for the paper-compatible RAG flow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newbaseline.src.rag.corpus import PaperEmbeddingCorpus
from newbaseline.src.rag.router import PAPER_SERIES, PaperNNRouter
from newbaseline.src.rag.service import PaperRagService
from newbaseline.src.rag.types import RetrievalHit
from newbaseline.src.rag.vocabulary import Vocabulary


class PaperEmbeddingCorpusTests(unittest.TestCase):
    def test_retrieval_uses_metadata_source_index_and_retains_empty_router_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunk_root = root / "chunks"
            chunk_root.mkdir()
            (chunk_root / "ChunkSeries21.json").write_text(
                json.dumps([
                    {"chunk_id": "original-0", "text": "less relevant"},
                    {"chunk_id": "original-1", "text": "most relevant"},
                ]),
                encoding="utf-8",
            )
            np.save(root / "series21.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
            (root / "series21.jsonl").write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        {"chunk_id": "original-0", "source_chunk_index": 0, "document_name": "a"},
                        {"chunk_id": "original-1", "source_chunk_index": 1, "document_name": "b"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_chunk_directory": str(chunk_root),
                        "series": {
                            "21": {
                                "vector_file": "series21.npy",
                                "metadata_file": "series21.jsonl",
                                "chunk_file": "ChunkSeries21.json",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            corpus = PaperEmbeddingCorpus(root)
            hits, searched, empty = corpus.search(["25", "21"], np.asarray([1.0, 0.0]), top_k=1)
        self.assertEqual(empty, ["25"])
        self.assertEqual(searched, ["21"])
        self.assertEqual(hits[0].text, "most relevant")
        self.assertEqual(hits[0].metadata["chunk_id"], "original-1")


class PaperRagServiceTests(unittest.TestCase):
    def test_service_exposes_selected_empty_and_searched_series(self) -> None:
        class FakeClient:
            def rephrase(self, question: str) -> str:
                return f"rephrased {question}"

            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0]

            def embed_many(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

            def answer(self, question: str, contexts: list[str]) -> str:
                return "answer"

        class FakeRouter:
            def route(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
                return ["25", "21"]

        class FakeCorpus:
            def search(self, selected: list[str], embedding: np.ndarray, top_k: int):
                hit = RetrievalHit(0.8, "21", "retrieved text", {"chunk_id": "original-1"})
                return [hit], ["21", "release-summaries"], ["25"]

            def expand_citations(self, seeds, **kwargs):
                return seeds, []

        service = object.__new__(PaperRagService)
        service.client = FakeClient()
        service.router = FakeRouter()
        service.corpus = FakeCorpus()
        service.vocabulary = Vocabulary({}, {})
        service.settings = type("Settings", (), {"get": lambda *_: 5})()
        result = service.run("What is AMF?", include_answer=False)
        self.assertEqual(result.router_selected_series, ["25", "21"])
        self.assertEqual(result.empty_selected_series, ["25"])
        self.assertEqual(result.searched_series, ["21", "release-summaries"])
        self.assertIsNone(result.answer)


class PaperRouterTests(unittest.TestCase):
    def test_checkpoint_accepts_1024_dimension_query_and_keeps_18_labels(self) -> None:
        root = Path(__file__).resolve().parents[2]
        router = PaperNNRouter(
            root / "newbaseline/resources/router_new.pth",
            root / "newbaseline/resources/series_description.json",
        )
        selected = router.route(np.zeros(1024, dtype=np.float32), top_k=5)
        self.assertEqual(len(selected), 5)
        self.assertTrue(set(selected).issubset(PAPER_SERIES))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
