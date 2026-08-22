"""Focused tests for contextual acronym disambiguation."""

from __future__ import annotations

import unittest

from newbaseline.src.rag.abbreviation_resolver import ContextualAbbreviationResolver
from newbaseline.src.rag.types import RetrievalHit
from newbaseline.src.rag.vocabulary import AmbiguousAbbreviationCandidate


class ContextualAbbreviationResolverTests(unittest.TestCase):
    def test_selects_a_clear_winner_and_records_evidence(self) -> None:
        resolver = ContextualAbbreviationResolver()
        matches = {
            "ARP": [
                AmbiguousAbbreviationCandidate("Allocation and Retention Priority", ("23",), 4),
                AmbiguousAbbreviationCandidate("Address Resolution Protocol", ("29",), 2),
            ]
        }
        hits = {
            ("ARP", "Allocation and Retention Priority"): [RetrievalHit(0.82, "23", "ARP", {"chunk_id": "a"})],
            ("ARP", "Address Resolution Protocol"): [RetrievalHit(0.58, "29", "ARP", {"chunk_id": "b"})],
        }

        result = resolver.resolve(matches, hits, [], ["23"], min_score=0.55, min_margin=0.015)

        self.assertEqual(result[0].selected_expansion, "Allocation and Retention Priority")
        self.assertGreater(result[0].confidence, 0.55)

    def test_abstains_when_two_senses_are_too_close(self) -> None:
        resolver = ContextualAbbreviationResolver()
        matches = {
            "AMF": [
                AmbiguousAbbreviationCandidate("Access and Mobility Management Function", ("23",), 1),
                AmbiguousAbbreviationCandidate("Authentication Management Field", ("24",), 1),
            ]
        }
        hits = {
            ("AMF", "Access and Mobility Management Function"): [RetrievalHit(0.71, "23", "one", {"chunk_id": "a"})],
            ("AMF", "Authentication Management Field"): [RetrievalHit(0.70, "24", "two", {"chunk_id": "b"})],
        }

        result = resolver.resolve(matches, hits, [], [], min_score=0.55, min_margin=0.015)

        self.assertIsNone(result[0].selected_expansion)
        self.assertLess(result[0].margin or 0, 0.015)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
