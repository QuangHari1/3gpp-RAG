from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newbaseline.src.rag.corpus import PaperEmbeddingCorpus
from newbaseline.src.rag.types import RetrievalHit


def chunk(chunk_id: str, document_id: str, heading: str, references: list[dict] | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "series": document_id[:2],
        "heading": heading,
        "chunk_index_in_heading": int(chunk_id[-1]) if chunk_id[-1].isdigit() else 0,
        "references": references or [],
        "text": f"text for {chunk_id}",
    }


class CitationExpansionTests(unittest.TestCase):
    def test_expands_precise_cross_series_citations_breadth_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = root / "chunks"
            chunks.mkdir()
            seed = chunk(
                "seed0",
                "21001",
                "1",
                [
                    {
                        "type": "external",
                        "document_id": "22001",
                        "target_heading": "2",
                        "target_series": "22",
                        "target_chunk_file": "ChunkSeries22.json",
                    },
                    {"type": "external", "document_id": "22001", "target_chunk_file": "ChunkSeries22.json"},
                    {
                        "type": "external",
                        "document_id": "22999",
                        "target_heading": "9",
                        "target_chunk_file": "ChunkSeries22.json",
                    },
                ],
            )
            (chunks / "ChunkSeries21.json").write_text(json.dumps({"chunks": [seed]}), encoding="utf-8")
            target_one = chunk(
                "target0",
                "22001",
                "2",
                [
                    {
                        "type": "external",
                        "document_id": "23001",
                        "target_heading": "3",
                        "target_series": "23",
                        "target_chunk_file": "ChunkSeries23.json",
                    }
                ],
            )
            target_two = chunk("target1", "22001", "2")
            (chunks / "ChunkSeries22.json").write_text(
                json.dumps({"chunks": [target_one, target_two]}), encoding="utf-8"
            )
            (chunks / "ChunkSeries23.json").write_text(
                json.dumps({"chunks": [chunk("depth2a", "23001", "3"), chunk("depth2b", "23001", "3")]}),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps({"source_chunk_directory": str(chunks), "series": {"21": {}}}), encoding="utf-8"
            )
            corpus = PaperEmbeddingCorpus(root)
            expanded, paths = corpus.expand_citations(
                [RetrievalHit(0.9, "21", seed["text"], {"chunk_id": "seed0"}, "ChunkSeries21.json")],
                max_depth=2,
                total_limit=18,
                chunks_per_heading=2,
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "target0", "target1", "depth2a", "depth2b"])
        self.assertEqual([hit.citation_depth for hit in expanded], [0, 1, 1, 2, 2])
        self.assertEqual([hit.metadata["source_chunk_index"] for hit in expanded[1:]], [0, 1, 0, 1])
        self.assertEqual(paths[0].status, "resolved")
        self.assertEqual(paths[0].target_chunk_ids, ["target0", "target1"])
        self.assertIn("target_heading_not_found", [path.status for path in paths])

    def test_total_limit_caps_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = root / "chunks"
            chunks.mkdir()
            seed = chunk(
                "seed0",
                "21001",
                "1",
                [{"type": "internal", "document_id": "21001", "target_heading": "2", "target_chunk_file": "ChunkSeries21.json"}],
            )
            targets = [chunk("target0", "21001", "2"), chunk("target1", "21001", "2")]
            (chunks / "ChunkSeries21.json").write_text(json.dumps({"chunks": [seed, *targets]}), encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps({"source_chunk_directory": str(chunks), "series": {"21": {}}}), encoding="utf-8"
            )
            expanded, _ = PaperEmbeddingCorpus(root).expand_citations(
                [RetrievalHit(0.9, "21", seed["text"], {"chunk_id": "seed0"}, "ChunkSeries21.json")],
                max_depth=2,
                total_limit=2,
                chunks_per_heading=2,
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "target0"])

    def test_ranks_oversized_cited_heading_by_query_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = root / "chunks"
            chunks.mkdir()
            seed = chunk(
                "seed0",
                "21001",
                "1",
                [{"type": "internal", "document_id": "21001", "target_heading": "2", "target_chunk_file": "ChunkSeries21.json"}],
            )
            targets = [chunk("target0", "21001", "2"), chunk("target1", "21001", "2"), chunk("target2", "21001", "2")]
            (chunks / "ChunkSeries21.json").write_text(json.dumps({"chunks": [seed, *targets]}), encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps({"source_chunk_directory": str(chunks), "series": {"21": {}}}), encoding="utf-8"
            )

            vectors = {
                "text for target0": [0.0, 1.0],
                "text for target1": [0.8, 0.0],
                "text for target2": [1.0, 0.0],
            }
            expanded, _ = PaperEmbeddingCorpus(root).expand_citations(
                [RetrievalHit(0.9, "21", seed["text"], {"chunk_id": "seed0"}, "ChunkSeries21.json")],
                max_depth=1,
                total_limit=3,
                chunks_per_heading=2,
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embed_many=lambda texts: [vectors[text] for text in texts],
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "target2", "target1"])
        self.assertAlmostEqual(expanded[1].score, 1.0)
        self.assertAlmostEqual(expanded[2].score, 0.8)

    def test_resolves_parent_heading_to_query_ranked_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = root / "chunks"
            chunks.mkdir()
            seed = chunk(
                "seed0",
                "21001",
                "1",
                [{"type": "internal", "document_id": "21001", "target_heading": "2", "target_chunk_file": "ChunkSeries21.json"}],
            )
            children = [
                chunk("child0", "21001", "2.1"),
                chunk("child1", "21001", "2.2"),
                chunk("child2", "21001", "2.3"),
            ]
            (chunks / "ChunkSeries21.json").write_text(
                json.dumps({"chunks": [seed, *children]}), encoding="utf-8"
            )
            (root / "manifest.json").write_text(
                json.dumps({"source_chunk_directory": str(chunks), "series": {"21": {}}}), encoding="utf-8"
            )
            vectors = {
                "text for child0": [0.2, 0.0],
                "text for child1": [0.9, 0.0],
                "text for child2": [1.0, 0.0],
            }
            expanded, paths = PaperEmbeddingCorpus(root).expand_citations(
                [RetrievalHit(0.9, "21", seed["text"], {"chunk_id": "seed0"}, "ChunkSeries21.json")],
                max_depth=1,
                total_limit=3,
                chunks_per_heading=2,
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embed_many=lambda texts: [vectors[text] for text in texts],
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "child2", "child1"])
        self.assertEqual(paths[0].status, "resolved_descendant")
        self.assertEqual(paths[0].target_chunk_ids, ["child2", "child1"])

    def test_depth_one_citations_are_selected_globally_not_by_reference_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = root / "chunks"
            chunks.mkdir()
            seed = chunk(
                "seed0",
                "21001",
                "1",
                [
                    {"type": "internal", "document_id": "21001", "target_heading": "2", "target_chunk_file": "ChunkSeries21.json"},
                    {"type": "internal", "document_id": "21001", "target_heading": "3", "target_chunk_file": "ChunkSeries21.json"},
                ],
            )
            earlier = chunk("earlier0", "21001", "2")
            better = chunk("better0", "21001", "3")
            (chunks / "ChunkSeries21.json").write_text(
                json.dumps({"chunks": [seed, earlier, better]}), encoding="utf-8"
            )
            (root / "manifest.json").write_text(
                json.dumps({"source_chunk_directory": str(chunks), "series": {"21": {}}}), encoding="utf-8"
            )

            vectors = {
                "text for earlier0": [0.1, 0.0],
                "text for better0": [1.0, 0.0],
            }
            expanded, paths = PaperEmbeddingCorpus(root).expand_citations(
                [RetrievalHit(0.9, "21", seed["text"], {"chunk_id": "seed0"}, "ChunkSeries21.json")],
                max_depth=1,
                total_limit=2,
                chunks_per_heading=2,
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embed_many=lambda texts: [vectors[text] for text in texts],
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "better0"])
        self.assertEqual(paths[0].status, "not_selected_by_query_score")
        self.assertEqual(paths[1].target_chunk_ids, ["better0"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
