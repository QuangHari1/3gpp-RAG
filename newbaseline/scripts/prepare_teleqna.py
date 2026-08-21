"""Download the 3GPP-only TeleQnA evaluation set.

The script writes one clean evaluation artifact to
``dataset/teleqna/TeleQnA.json``. The unfiltered upstream source and Hugging
Face metadata are kept in a temporary directory, not in the repository.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.settings import load_settings

SETTINGS = load_settings()
DATASET_REVISION = SETTINGS.get("teleqna", "revision")
THREE_GPP_RELEASE_TAG = re.compile(SETTINGS.get("teleqna", "release_tag_pattern"))
WORKSPACE_ROOT = SETTINGS.workspace_root
DATASET_DIR = SETTINGS.dataset_dir / "teleqna"
OUTPUT_PATH = DATASET_DIR / SETTINGS.get("teleqna", "include_file")


def download_records() -> dict[str, dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="teleqna-download-") as download_dir:
        subprocess.run(
            [
                "hf",
                "download",
                SETTINGS.get("teleqna", "repository"),
                "--type",
                "dataset",
                "--revision",
                DATASET_REVISION,
                "--include",
                SETTINGS.get("teleqna", "include_file"),
                "--local-dir",
                download_dir,
                "--format",
                "quiet",
            ],
            check=True,
        )
        records = json.loads(
            (Path(download_dir) / SETTINGS.get("teleqna", "include_file")).read_text(encoding="utf-8")
        )

    if not isinstance(records, dict):
        raise ValueError("Expected TeleQnA.json to be a mapping of question IDs to records.")
    return records


def is_3gpp_release_question(record: dict[str, object]) -> bool:
    question = record.get("question")
    return isinstance(question, str) and bool(THREE_GPP_RELEASE_TAG.search(question))


def prepare_teleqna_subset() -> int:
    records = download_records()
    filtered = {
        question_id: record
        for question_id, record in records.items()
        if isinstance(record, dict) and is_3gpp_release_question(record)
    }

    DATASET_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=DATASET_DIR,
        delete=False,
    ) as temporary_file:
        json.dump(filtered, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)

    temporary_path.replace(OUTPUT_PATH)
    for path in DATASET_DIR.iterdir():
        if path == OUTPUT_PATH:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    return len(filtered)


if __name__ == "__main__":
    total = prepare_teleqna_subset()
    print(f"Prepared {total} 3GPP Release questions at {OUTPUT_PATH}")
