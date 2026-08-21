"""Optional experiment tracking for reproducible benchmark runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


VALID_WANDB_MODES = {"disabled", "offline", "online"}


@dataclass
class ExperimentTracker:
    """Log benchmark summaries and durable result artifacts to W&B when enabled."""

    run: Any | None = None
    wandb: Any | None = None
    artifact_name: str = "teleqna-benchmark"

    @property
    def enabled(self) -> bool:
        return self.run is not None and self.wandb is not None

    def log_progress(self, completed: int, correct: int, evaluated: int) -> None:
        if not self.enabled:
            return
        self.run.log(
            {
                "benchmark/completed": completed,
                "benchmark/correct": correct,
                "benchmark/evaluated": evaluated,
                "benchmark/accuracy": correct / evaluated if evaluated else 0.0,
            },
            step=completed,
        )

    def log_comparison(self, comparison: dict[str, Any]) -> None:
        """Attach a same-question benchmark comparison to the active run."""
        if not self.enabled:
            return
        self.run.summary.update({f"comparison/{key}": value for key, value in comparison.items()})

    def finish(
        self,
        *,
        status: str,
        completed: int,
        correct: int,
        evaluated: int,
        results_path: Path,
        manifest_path: Path,
        analysis_directory: Path | None = None,
        comparison_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        summary = {
            "benchmark/status": status,
            "benchmark/completed": completed,
            "benchmark/correct": correct,
            "benchmark/evaluated": evaluated,
            "benchmark/accuracy": correct / evaluated if evaluated else 0.0,
        }
        if error:
            summary["benchmark/error"] = error
        self.run.summary.update(summary)
        if results_path.exists():
            artifact = self.wandb.Artifact(name=self.artifact_name, type="benchmark-results")
            artifact.add_file(str(results_path), name=results_path.name)
            if manifest_path.exists():
                artifact.add_file(str(manifest_path), name=manifest_path.name)
            if comparison_path and comparison_path.is_file():
                artifact.add_file(str(comparison_path), name=comparison_path.name)
            analysis_summary = analysis_directory / "summary.md" if analysis_directory else None
            if (
                analysis_directory
                and analysis_summary
                and analysis_summary.is_file()
                and analysis_summary.stat().st_mtime >= results_path.stat().st_mtime
            ):
                for analysis_file in sorted(analysis_directory.iterdir()):
                    if analysis_file.is_file() and analysis_file.suffix in {".csv", ".json", ".md"}:
                        artifact.add_file(str(analysis_file), name=f"analysis/{analysis_file.name}")
            self.run.log_artifact(artifact)
        self.run.finish()


def start_experiment_tracker(
    *,
    mode: str,
    project: str,
    entity: str,
    artifact_name: str,
    run_directory: Path,
    output_path: Path,
    config: dict[str, Any],
    wandb_module: Any | None = None,
) -> ExperimentTracker:
    """Create a W&B run only when tracking is deliberately enabled."""
    if mode not in VALID_WANDB_MODES:
        raise ValueError(f"Unsupported experiment tracking mode: {mode}")
    if mode == "disabled":
        return ExperimentTracker()
    wandb = wandb_module or import_module("wandb")
    run_directory.mkdir(parents=True, exist_ok=True)
    init_options = {
        "project": project,
        "entity": entity or None,
        "mode": mode,
        "dir": str(run_directory),
        "name": output_path.stem,
        "config": config,
        "save_code": False,
    }
    if mode == "online":
        run_id = hashlib.sha256(str(output_path.resolve()).encode("utf-8")).hexdigest()[:16]
        init_options.update({"id": run_id, "resume": "allow"})
    run = wandb.init(
        **init_options,
    )
    return ExperimentTracker(run=run, wandb=wandb, artifact_name=artifact_name)
