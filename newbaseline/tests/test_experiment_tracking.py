from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from newbaseline.src.evaluation.tracking import scalar_params, start_experiment_tracker


class ExperimentTrackingTests(unittest.TestCase):
    def test_disabled_mode_does_not_create_mlflow_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracker = start_experiment_tracker(
                mode="disabled",
                experiment_name="telco-rag",
                tracking_directory=root / "mlflow",
                output_path=root / "results.jsonl",
                config={},
            )
        self.assertFalse(tracker.enabled)
        self.assertFalse((root / "mlflow").exists())

    def test_mlflow_mode_logs_progress_metrics_and_artifacts(self) -> None:
        class FakeMlflow:
            def __init__(self) -> None:
                self.tracking_uri: str | None = None
                self.created_experiments: list[tuple[str, str]] = []
                self.experiment_name: str | None = None
                self.params: dict = {}
                self.tags: list[dict] = []
                self.metrics: list[tuple[dict, int | None]] = []
                self.artifacts: list[tuple[str, str | None]] = []
                self.ended_status: str | None = None

            def set_tracking_uri(self, uri: str) -> None:
                self.tracking_uri = uri

            def get_experiment_by_name(self, name: str):
                return None

            def create_experiment(self, name: str, artifact_location: str) -> str:
                self.created_experiments.append((name, artifact_location))
                return "1"

            def set_experiment(self, name: str) -> None:
                self.experiment_name = name

            def start_run(self, run_name: str):
                self.run_name = run_name
                return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

            def log_params(self, values: dict) -> None:
                self.params.update(values)

            def set_tags(self, values: dict) -> None:
                self.tags.append(values)

            def log_metrics(self, values: dict, step: int | None = None) -> None:
                self.metrics.append((values, step))

            def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
                self.artifacts.append((Path(path).name, artifact_path))

            def end_run(self, status: str) -> None:
                self.ended_status = status

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = root / "results.jsonl"
            manifest = root / "results.manifest.json"
            comparison = root / "results.comparison.json"
            analysis_directory = root / "analysis"
            result.write_text("{}\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            comparison.write_text("{}\n", encoding="utf-8")
            analysis_directory.mkdir()
            (analysis_directory / "summary.md").write_text("# summary\n", encoding="utf-8")
            fake_mlflow = FakeMlflow()
            tracker = start_experiment_tracker(
                mode="mlflow",
                experiment_name="telco-rag",
                tracking_directory=root / "mlflow",
                output_path=result,
                config={"citation_total_chunks": 13, "excluded": ["3GPP"]},
                mlflow_module=fake_mlflow,
                source_fingerprint="abc",
            )
            tracker.log_progress(completed=10, correct=7, evaluated=10)
            tracker.log_comparison({"shared_scored_questions": 10, "accuracy_delta": 0.1, "candidate_path": "x"})
            tracker.finish(
                status="completed",
                completed=10,
                correct=7,
                evaluated=10,
                results_path=result,
                manifest_path=manifest,
                analysis_directory=analysis_directory,
                comparison_path=comparison,
            )

        self.assertEqual(fake_mlflow.experiment_name, "telco-rag")
        self.assertEqual(fake_mlflow.run_name, "results")
        self.assertEqual(fake_mlflow.params["excluded"], '["3GPP"]')
        self.assertEqual(fake_mlflow.metrics[0][0]["benchmark.accuracy"], 0.7)
        self.assertTrue(any(tag.get("benchmark.status") == "completed" for tag in fake_mlflow.tags))
        self.assertTrue(any(tag.get("comparison.candidate_path") == "x" for tag in fake_mlflow.tags))
        self.assertEqual(
            fake_mlflow.artifacts,
            [
                ("results.jsonl", None),
                ("results.manifest.json", None),
                ("results.comparison.json", None),
                ("summary.md", "analysis"),
            ],
        )
        self.assertEqual(fake_mlflow.ended_status, "FINISHED")

    def test_scalar_params_serializes_nested_values(self) -> None:
        self.assertEqual(scalar_params({"items": ["a"], "none": None}), {"items": '["a"]', "none": "null"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
