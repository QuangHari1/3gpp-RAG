"""Stable document discovery for either original or verbalized Markdown."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    """The chosen source file for one 3GPP document directory."""

    path: Path
    document_key: str
    source_name: str


def discover_source_documents(
    release_dir: Path,
    source_names: tuple[str, ...],
    selected_document_keys: set[str] | None = None,
) -> list[SourceDocument]:
    """Choose one Markdown source per document, using priority order.

    A verbalized-only corpus is valid: discovery takes the union of all source
    names, rather than assuming every directory first contains ``raw.md``.
    """
    if not source_names:
        raise ValueError("At least one source Markdown filename is required.")
    candidates: dict[Path, dict[str, Path]] = {}
    for source_name in source_names:
        for path in release_dir.rglob(source_name):
            document_dir = path.parent
            candidates.setdefault(document_dir, {})[source_name] = path

    documents: list[SourceDocument] = []
    for document_dir, by_name in candidates.items():
        selected_name = next((name for name in source_names if name in by_name), None)
        if selected_name is None:  # defensive: impossible after construction
            continue
        document_key = document_dir.relative_to(release_dir).as_posix()
        if selected_document_keys is not None and document_key not in selected_document_keys:
            continue
        documents.append(SourceDocument(by_name[selected_name], document_key, selected_name))
    return sorted(documents, key=lambda item: item.document_key)


def load_selected_document_keys(selection_path: Path | None) -> set[str] | None:
    """Read the shared selection format used by chunking and embedding."""
    if selection_path is None:
        return None
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"Selection has no documents list: {selection_path}")
    keys = {
        document["document_key"]
        for document in documents
        if isinstance(document, dict) and isinstance(document.get("document_key"), str)
    }
    if not keys:
        raise ValueError(f"Selection has no document keys: {selection_path}")
    return keys
