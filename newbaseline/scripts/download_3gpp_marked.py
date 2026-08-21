"""Download the text-only Release 18 Markdown corpus from GSMA/3GPP.

Only ``raw.md`` files under upstream ``marked/Rel-18/`` are downloaded. The
original source documents, other releases, extracted figures, and Hugging Face
local metadata are excluded from the final output.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.settings import load_settings

SETTINGS = load_settings()
DATASET_REPOSITORY = SETTINGS.get("gsma", "repository")
DATASET_REVISION = SETTINGS.get("gsma", "revision")
WORKSPACE_ROOT = SETTINGS.workspace_root
DATASET_DIR = SETTINGS.dataset_dir / "3gpp"
MARKED_DIR = DATASET_DIR / "marked"
RELEASE_DIR = SETTINGS.release_dir
RAW_MARKDOWN_GLOB = SETTINGS.get("gsma", "raw_markdown_glob")


def remove_huggingface_metadata() -> None:
    """Remove the local-dir metadata written by ``hf download``."""
    metadata_dir = DATASET_DIR / ".cache"
    if metadata_dir.exists():
        shutil.rmtree(metadata_dir)


def remove_extracted_figures() -> None:
    """Keep only raw Markdown files if an earlier download included figures."""
    if not RELEASE_DIR.exists():
        return

    for path in RELEASE_DIR.rglob("*"):
        if path.is_file() and path.name != "raw.md":
            path.unlink()


def count_markdown_documents() -> int:
    if not RELEASE_DIR.exists():
        return 0
    return sum(1 for _ in RELEASE_DIR.rglob("raw.md"))


def download_release_18_corpus() -> int:
    """Download Release 18 text documents, preserving Hub resume support."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Resuming Release 18 raw Markdown download at {RELEASE_DIR}", flush=True)
    command = [
        "hf",
        "download",
        DATASET_REPOSITORY,
        "--type",
        "dataset",
        "--revision",
        DATASET_REVISION,
        "--include",
        RAW_MARKDOWN_GLOB,
        "--local-dir",
        str(DATASET_DIR),
        "--format",
        "human",
    ]
    process = subprocess.Popen(command)
    while process.poll() is None:
        expected = SETTINGS.get("gsma", "expected_raw_markdown_files")
        print(f"Downloaded {count_markdown_documents()}/{expected} raw Markdown files...", flush=True)
        time.sleep(SETTINGS.get("runtime", "download_progress_seconds"))
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

    remove_extracted_figures()
    remove_huggingface_metadata()

    markdown_documents = count_markdown_documents()
    if markdown_documents == 0:
        raise RuntimeError(f"No Markdown documents were downloaded to {RELEASE_DIR}")
    return markdown_documents


if __name__ == "__main__":
    total = download_release_18_corpus()
    print(f"Downloaded {total} Release 18 raw Markdown documents to {RELEASE_DIR}")
