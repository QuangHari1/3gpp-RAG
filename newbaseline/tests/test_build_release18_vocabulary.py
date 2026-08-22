"""Focused tests for the Release-18 vocabulary catalog builder."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/build_release18_vocabulary.py"
SPEC = importlib.util.spec_from_file_location("build_release18_vocabulary", SCRIPT_PATH)
assert SPEC and SPEC.loader
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class BuildRelease18VocabularyTests(unittest.TestCase):
    def write_document(self, root: Path, document: str, text: str) -> None:
        path = root / "23_series" / document / "raw.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_keeps_distinct_meanings_and_merges_identical_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_document(
                root,
                "23501",
                """## 3 Abbreviations
| AMF | Access and Mobility Management Function |
| ARP | Allocation and Retention Priority |
""",
            )
            self.write_document(
                root,
                "23502",
                """## 3 Abbreviations
| AMF | Access and Mobility Management Function |
| ARP | Address Resolution Protocol |
""",
            )
            catalog = BUILD.build_catalog(root, "18")

        self.assertEqual(catalog["acronyms"]["AMF"][0]["expansion"], "Access and Mobility Management Function")
        self.assertEqual(len(catalog["acronyms"]["AMF"][0]["sources"]), 2)
        self.assertEqual(
            [candidate["expansion"] for candidate in catalog["acronyms"]["ARP"]],
            ["Address Resolution Protocol", "Allocation and Retention Priority"],
        )
        self.assertEqual(catalog["ambiguous_acronyms"], 1)

    def test_ignores_tables_outside_abbreviations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_document(
                root,
                "23501",
                """## 3 Definitions
| AMF | Incorrect table row |

## 4 Abbreviations
| AMF | Access and Mobility Management Function |
""",
            )
            catalog = BUILD.build_catalog(root, "18")

        self.assertEqual(
            catalog["acronyms"]["AMF"][0]["expansion"], "Access and Mobility Management Function"
        )

    def test_stops_at_the_next_peer_heading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_document(
                root,
                "23501",
                """## 3 Abbreviations
| AMF | Access and Mobility Management Function |

## 4 Technical table
| AMF | A sentence about a field in this technical table |
""",
            )
            catalog = BUILD.build_catalog(root, "18")

        self.assertEqual(
            catalog["acronyms"]["AMF"][0]["expansion"], "Access and Mobility Management Function"
        )


if __name__ == "__main__":
    unittest.main()
