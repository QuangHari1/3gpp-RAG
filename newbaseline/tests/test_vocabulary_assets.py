"""Tests for the opt-in two-asset vocabulary mode."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newbaseline.src.rag.vocabulary import AmbiguousAbbreviationCandidate, Vocabulary


class VocabularyAssetsTests(unittest.TestCase):
    def test_release18_assets_keep_terms_and_abstain_for_ambiguous_acronyms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            definitions = root / "definitions.json"
            abbreviations = root / "abbreviations.json"
            definitions.write_text(
                json.dumps({"terms": [{"term": "PDU Session", "definition": "A session definition"}]}),
                encoding="utf-8",
            )
            abbreviations.write_text(
                json.dumps(
                    {
                        "acronyms": {
                            "SMF": [{"expansion": "Session Management Function"}],
                            "AMF": [
                                {"expansion": "Access and Mobility Management Function"},
                                {"expansion": "Authentication Management Field"},
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            vocabulary = Vocabulary.from_release18_assets(definitions, abbreviations)

        enriched = vocabulary.enrich("Does AMF coordinate a PDU Session with SMF?")
        self.assertIn("PDU Session: A session definition", enriched)
        self.assertIn("SMF: Session Management Function", enriched)
        self.assertNotIn("AMF:", enriched)

    def test_excluded_metadata_acronym_does_not_trigger_contextual_resolution(self) -> None:
        vocabulary = Vocabulary(
            {},
            {},
            {
                "3GPP": [
                    AmbiguousAbbreviationCandidate("Third Generation Partnership Project", ("33",), 1),
                    AmbiguousAbbreviationCandidate("3rd Generation Partnership Project", ("33",), 1),
                ],
                "AMF": [
                    AmbiguousAbbreviationCandidate("Access and Mobility Management Function", ("23",), 1),
                    AmbiguousAbbreviationCandidate("Authentication Management Field", ("24",), 1),
                ],
            },
        )

        matches = vocabulary.ambiguous_matches("[3GPP Release 18] What is AMF?", 8, ["3gpp"])

        self.assertEqual(list(matches), ["AMF"])


if __name__ == "__main__":
    unittest.main()
