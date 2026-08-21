from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from newbaseline.src.evaluation.tracking import start_experiment_tracker


class ExperimentTrackingTests(unittest.TestCase):
    def test_disabled_mode_does_not_import_or_write_wandb_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracker = start_experiment_tracker(
                mode="disabled",
                project="telco-rag",
                entity="",
                artifact_name="teleqna-benchmark",
                run_directory=root / "wandb",
                output_path=root / "results.jsonl",
                config={},
            )
        self.assertFalse(tracker.enabled)
        self.assertFalse((root / "wandb").exists())

    def test_offline_mode_logs_progress_and_result_artifact(self) -> None:
        class FakeArtifact:
            def __init__(self, name: str, type: str) -> None:
                self.name = name
                self.type = type
                self.files: list[tuple[str, str]] = []

            def add_file(self, path: str, name: str) -> None:
                self.files.append((path, name))

        class FakeRun:
            def __init__(self) -> None:
                self.logs: list[tuple[dict, int]] = []
                self.summary: dict = {}
                self.artifacts: list[FakeArtifact] = []
                self.finished = False

            def log(self, values: dict, step: int) -> None:
                self.logs.append((values, step))

            def log_artifact(self, artifact: FakeArtifact) -> None:
                self.artifacts.append(artifact)

            def finish(self) -> None:
                self.finished = True

        class FakeWandb:
            def __init__(self) -> None:
                self.run = FakeRun()
                self.init_kwargs: dict | None = None

            def init(self, **kwargs):
                self.init_kwargs = kwargs
                return self.run

            Artifact = FakeArtifact

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
            fake_wandb = FakeWandb()
            tracker = start_experiment_tracker(
                mode="offline",
                project="telco-rag",
                entity="",
                artifact_name="teleqna-benchmark",
                run_directory=root / "wandb",
                output_path=result,
                config={"citation_total_chunks": 13},
                wandb_module=fake_wandb,
            )
            tracker.log_progress(completed=10, correct=7, evaluated=10)
            tracker.log_comparison({"shared_scored_questions": 10, "accuracy_delta": 0.1})
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

        self.assertEqual(fake_wandb.init_kwargs["mode"], "offline")
        self.assertNotIn("resume", fake_wandb.init_kwargs)
        self.assertEqual(fake_wandb.run.logs[0][0]["benchmark/accuracy"], 0.7)
        self.assertEqual(fake_wandb.run.summary["benchmark/status"], "completed")
        self.assertEqual(fake_wandb.run.summary["comparison/accuracy_delta"], 0.1)
        self.assertEqual(
            [name for _, name in fake_wandb.run.artifacts[0].files],
            ["results.jsonl", "results.manifest.json", "results.comparison.json", "analysis/summary.md"],
        )
        self.assertTrue(fake_wandb.run.finished)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
