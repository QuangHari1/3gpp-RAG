"""Paper-compatible 3GPP terminology and abbreviation enrichment."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document


class Vocabulary:
    def __init__(self, terms: dict[str, str], abbreviations: dict[str, str]):
        self.terms = terms
        self.abbreviations = abbreviations

    @classmethod
    def from_docx(cls, path: Path) -> "Vocabulary":
        terms: dict[str, str] = {}
        abbreviations: dict[str, str] = {}
        in_terms = False
        in_abbreviations = False
        reference_count = 0
        for paragraph in Document(path).paragraphs:
            text = paragraph.text.strip()
            if "References" in text:
                reference_count += 1
            if reference_count < 2:
                continue
            if "Terms and definitions" in text:
                in_terms, in_abbreviations = True, False
            elif "Abbreviations" in text:
                in_terms, in_abbreviations = False, True
            elif in_terms and ":" in text:
                term, definition = text.split(":", 1)
                terms[term.strip()] = definition.strip().rstrip(".")
            elif in_abbreviations and "\t" in text:
                abbreviation, definition = text.split("\t", 1)
                if len(abbreviation.strip()) > 1:
                    abbreviations[abbreviation.strip()] = definition.strip()
        return cls(terms, abbreviations)

    def enrich(self, question: str) -> str:
        normalized = re.sub(r"[!()\-\[\]{};:'\"\\,<>./?@#$%^&*_~]", "", question.lower())
        matched_terms = [
            f"{term}: {definition}"
            for term, definition in self.terms.items()
            if re.sub(r"[!()\-\[\]{};:'\"\\,<>./?@#$%^&*_~]", "", term.lower()) in normalized
        ]
        words = re.sub(r"[!()\-\[\]{};:'\"\\,<>./?@#$%^&*_~]", "", question).split()
        matched_abbreviations = [
            f"{word}: {self.abbreviations[word]}" for word in words if word in self.abbreviations
        ]
        terms_text = "\n".join(matched_terms)
        abbreviations_text = "\n".join(matched_abbreviations)
        return (
            f"{question}\n\nTerms and Definitions:\n\n{terms_text}"
            f"\n\nAbbreviations:\n\n{abbreviations_text}\n"
        )
