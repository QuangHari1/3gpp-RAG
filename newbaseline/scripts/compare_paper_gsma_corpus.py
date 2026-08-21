"""Map the paper corpus filenames to local GSMA Release 18 Markdown paths.

This script only reads metadata already stored locally.  It never downloads the
paper corpus or its precomputed embeddings.  The paper names each DOCX as
``<spec-id>-<paper-version>.docx`` while GSMA stores text below
``<series>_series/<spec-id-or-variant>/raw.md``.  The shared five-digit
specification ID is the comparison key; every GSMA version variant is retained
in the report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.settings import load_settings

SETTINGS = load_settings()
WORKSPACE_ROOT = SETTINGS.workspace_root
DEFAULT_PAPER_MANIFEST = SETTINGS.dataset_dir / "3gpp/paper_baseline_documents.json"
DEFAULT_GSMA_RELEASE_DIR = SETTINGS.release_dir
DEFAULT_OUTPUT = SETTINGS.dataset_dir / "3gpp/paper_gsma_rel18_mapping.json"
PAPER_FILENAME_PATTERN = re.compile(r"(?P<spec_id>\d{5})-(?P<version>.+)\.docx$")
GSMA_DIRECTORY_PATTERN = re.compile(r"(?P<spec_id>\d{5})(?:-.+)?$")


def load_paper_documents(manifest_path: Path) -> list[dict[str, object]]:
    """Return the paper's DOCX metadata, rejecting unexpected entries."""
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"Missing documents list in {manifest_path}")
    return documents


def collect_gsma_documents(release_dir: Path) -> dict[str, list[str]]:
    """Index GSMA raw Markdown paths by their five-digit specification ID."""
    if not release_dir.is_dir():
        raise FileNotFoundError(f"GSMA Release 18 corpus not found: {release_dir}")

    documents: dict[str, list[str]] = {}
    for raw_path in sorted(release_dir.rglob("raw.md")):
        directory_match = GSMA_DIRECTORY_PATTERN.fullmatch(raw_path.parent.name)
        if directory_match is None:
            raise ValueError(f"Unexpected GSMA document directory: {raw_path.parent}")
        specification_id = directory_match.group("spec_id")
        try:
            document_path = str(raw_path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            document_path = str(raw_path)
        documents.setdefault(specification_id, []).append(document_path)
    return documents


def build_mapping(
    paper_documents: list[dict[str, object]], gsma_documents: dict[str, list[str]]
) -> dict[str, object]:
    """Build a filename-aware comparison report without inspecting document text."""
    rows: list[dict[str, object]] = []
    paper_specification_ids: set[str] = set()

    for document in paper_documents:
        kind = document.get("kind")
        paper_path = document.get("path")
        if not isinstance(kind, str) or not isinstance(paper_path, str):
            raise ValueError(f"Malformed paper document metadata: {document}")

        filename = Path(paper_path).name
        if kind == "release_summary":
            rows.append(
                {
                    "kind": kind,
                    "paper_path": paper_path,
                    "paper_filename": filename,
                    "canonical_spec_id": None,
                    "gsma_raw_markdown_paths": [],
                    "match": "not_a_specification",
                }
            )
            continue

        if kind != "specification":
            raise ValueError(f"Unexpected paper document kind: {kind}")

        match = PAPER_FILENAME_PATTERN.fullmatch(filename)
        if match is None:
            raise ValueError(f"Unexpected paper specification filename: {filename}")

        specification_id = match.group("spec_id")
        paper_specification_ids.add(specification_id)
        gsma_paths = gsma_documents.get(specification_id, [])
        rows.append(
            {
                "kind": kind,
                "paper_path": paper_path,
                "paper_filename": filename,
                "paper_version_suffix": match.group("version"),
                "canonical_spec_id": specification_id,
                "gsma_raw_markdown_paths": gsma_paths,
                "match": "shared_specification_id" if gsma_paths else "paper_only_specification_id",
            }
        )

    shared_ids = paper_specification_ids & gsma_documents.keys()
    return {
        "mapping_rule": "paper filename first five digits == GSMA raw.md parent directory first five digits",
        "paper_document_files": len(paper_documents),
        "paper_unique_specification_ids": len(paper_specification_ids),
        "gsma_raw_markdown_files": sum(len(paths) for paths in gsma_documents.values()),
        "gsma_unique_specification_ids": len(gsma_documents),
        "shared_specification_ids": len(shared_ids),
        "paper_only_specification_ids": len(paper_specification_ids - gsma_documents.keys()),
        "gsma_only_specification_ids": len(gsma_documents.keys() - paper_specification_ids),
        "paper_coverage_of_gsma_unique_specification_ids": round(
            len(shared_ids) / len(gsma_documents), 6
        ) if gsma_documents else 0.0,
        "documents": rows,
    }


def write_report(output_path: Path, report: dict[str, object]) -> None:
    """Atomically write the reproducible metadata report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        json.dump(report, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER_MANIFEST)
    parser.add_argument("--gsma-release-dir", type=Path, default=DEFAULT_GSMA_RELEASE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_mapping(
        load_paper_documents(args.paper_manifest),
        collect_gsma_documents(args.gsma_release_dir),
    )
    write_report(args.output, report)
    print(
        "Wrote mapping: "
        f"{report['shared_specification_ids']}/{report['gsma_raw_markdown_files']} shared GSMA IDs "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
