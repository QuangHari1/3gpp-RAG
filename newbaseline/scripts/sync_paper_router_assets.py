"""Copy fixed paper router resources into newbaseline's owned resource area."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

FILES = ("router_new.pth", "series_description.json", "3GPP_vocabulary.docx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite resources whose content differs.")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    source = repository_root / "Telco-RAG_api" / "src" / "resources"
    destination = repository_root / "newbaseline" / "resources"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        source_path = source / filename
        destination_path = destination / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        same_content = destination_path.exists() and sha256(destination_path) == sha256(source_path)
        if destination_path.exists() and not same_content and not args.force:
            raise FileExistsError(f"{destination_path} differs; rerun with --force to replace it.")
        if not same_content:
            shutil.copy2(source_path, destination_path)
        print(f"{filename}: {sha256(destination_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
