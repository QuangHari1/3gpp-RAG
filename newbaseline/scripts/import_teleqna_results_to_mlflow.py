"""Import existing TeleQnA JSONL checkpoints and analysis artifacts into local MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from newbaseline.src.evaluation.tracking import start_experiment_tracker


DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "newbaseline/results/teleqna"
DEFAULT_ANALYSIS_DIRECTORY = PROJECT_ROOT / "newbaseline/results/analysis"
DEFAULT_TRACKING_DIRECTORY = PROJECT_ROOT / "newbaseline/results/mlflow"


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSONL row in {path}")
        rows.append(row)
    return rows


def source_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_import_index(path: Path) -> dict[str, dict[str, str]]:
    payload = load_json(path)
    entries = payload.get("imports")
    return entries if isinstance(entries, dict) else {}


def save_import_index(path: Path, entries: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"imports": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_results(
    results_directory: Path,
    analysis_directory: Path,
    tracking_directory: Path,
    experiment_name: str,
    force: bool = False,
) -> tuple[int, int]:
    """Import each result checkpoint once, keyed by its content fingerprint."""
    index_path = tracking_directory / "imported_sources.json"
    index = load_import_index(index_path)
    imported = skipped = 0
    for results_path in sorted(results_directory.glob("*.jsonl")):
        manifest_path = results_path.with_suffix(".manifest.json")
        comparison_path = results_path.with_suffix(".comparison.json")
        run_analysis_directory = analysis_directory / results_path.stem
        analysis_paths = (
            sorted(
                path
                for path in run_analysis_directory.iterdir()
                if path.is_file() and path.suffix in {".csv", ".json", ".md"}
            )
            if run_analysis_directory.is_dir()
            else []
        )
        source_paths = [
            results_path,
            *[path for path in (manifest_path, comparison_path) if path.is_file()],
            *analysis_paths,
        ]
        fingerprint = source_fingerprint(source_paths)
        index_key = str(results_path.resolve())
        if not force and index.get(index_key, {}).get("fingerprint") == fingerprint:
            skipped += 1
            continue
        rows = load_rows(results_path)
        scored = [row for row in rows if row.get("is_correct") is not None]
        correct = sum(row.get("is_correct") is True for row in scored)
        config = load_json(manifest_path)
        config["migration.source_jsonl"] = results_path.name
        config["migration.source_fingerprint"] = fingerprint
        tracker = start_experiment_tracker(
            mode="mlflow",
            experiment_name=experiment_name,
            tracking_directory=tracking_directory,
            output_path=results_path,
            config=config,
            source_fingerprint=fingerprint,
        )
        comparison = load_json(comparison_path)
        if comparison:
            tracker.log_comparison(comparison)
        run_id = getattr(getattr(tracker.run, "info", None), "run_id", "")
        tracker.finish(
            status="imported",
            completed=len(rows),
            correct=correct,
            evaluated=len(scored),
            results_path=results_path,
            manifest_path=manifest_path,
            analysis_directory=run_analysis_directory,
            comparison_path=comparison_path,
        )
        index[index_key] = {"fingerprint": fingerprint, "run_id": str(run_id)}
        save_import_index(index_path, index)
        imported += 1
    return imported, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIRECTORY)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIRECTORY)
    parser.add_argument("--experiment-name", default="telco-rag")
    parser.add_argument("--force", action="store_true", help="Import unchanged checkpoints again as new runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imported, skipped = import_results(
        args.results_dir, args.analysis_dir, args.tracking_dir, args.experiment_name, args.force
    )
    print(f"MLflow import complete: imported={imported}, unchanged_skipped={skipped}, store={args.tracking_dir}")


if __name__ == "__main__":
    main()
