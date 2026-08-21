"""Export the document list used by the Telco-oRAG paper baseline.

This reads only Hugging Face repository metadata; it does not download the
paper corpus or its precomputed embeddings.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from urllib.request import urlopen

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.settings import load_settings

SETTINGS = load_settings()
REPOSITORY = SETTINGS.get("paper", "repository")
REVISION = SETTINGS.get("paper", "revision")
EXPECTED_DOCUMENT_COUNT = SETTINGS.get("paper", "expected_document_files")
WORKSPACE_ROOT = SETTINGS.workspace_root
OUTPUT_PATH = SETTINGS.dataset_dir / "3gpp" / "paper_baseline_documents.json"
DOCUMENT_PATTERN = re.compile(r"Documents/(?P<document_id>\d{5})-(?P<version>[^/]+)\.docx$")
RELEASE_SUMMARY_PATTERN = re.compile(r"Documents/rel_(?P<release>\d+)\.docx$")


def list_document_paths() -> list[str]:
    query = urlencode({"recursive": "true", "expand": "false", "limit": 1000})
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/Documents?{query}"
    with urlopen(url, timeout=SETTINGS.get("runtime", "http_timeout_seconds")) as response:
        entries = json.load(response)

    paths = sorted(
        entry["path"]
        for entry in entries
        if entry.get("type") == "file" and isinstance(entry.get("path"), str)
    )
    if len(paths) != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DOCUMENT_COUNT} paper documents, found {len(paths)} at {REVISION}"
        )
    return paths


def write_list() -> int:
    documents = []
    for path in list_document_paths():
        match = DOCUMENT_PATTERN.fullmatch(path)
        if match:
            documents.append(
                {
                    "kind": "specification",
                    "document_id": match.group("document_id"),
                    "version": match.group("version"),
                    "path": path,
                }
            )
            continue

        summary_match = RELEASE_SUMMARY_PATTERN.fullmatch(path)
        if summary_match:
            documents.append(
                {
                    "kind": "release_summary",
                    "release": summary_match.group("release"),
                    "path": path,
                }
            )
            continue

        raise ValueError(f"Unexpected paper document path: {path}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=OUTPUT_PATH.parent,
        delete=False,
    ) as temporary_file:
        json.dump(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "document_count": len(documents),
                "documents": documents,
            },
            temporary_file,
            ensure_ascii=False,
            indent=2,
        )
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(OUTPUT_PATH)
    return len(documents)


if __name__ == "__main__":
    total = write_list()
    print(f"Wrote {total} paper corpus documents to {OUTPUT_PATH}")
