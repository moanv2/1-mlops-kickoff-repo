"""Tests for src.main module."""

import subprocess
import sys


class TestMainPipeline:
    """Tests for the main.py pipeline orchestrator."""

    def test_pipeline_runs_end_to_end(self):
        """The full pipeline should complete without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Pipeline failed with stderr:\n{result.stderr}"
        )

    def test_pipeline_outputs_metric(self):
        """The pipeline should print the final metric."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "Pipeline complete" in result.stdout
        assert "metric" in result.stdout

    def test_pipeline_outputs_predictions(self):
        """The pipeline should print prediction results."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "prediction" in result.stdout

    def test_pipeline_produces_model_artifact(self, tmp_path):
        """The pipeline should save a model file."""
        import os
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert os.path.exists("models/model.pkl")

    def test_pipeline_produces_confusion_matrix(self):
        """The pipeline should save a confusion matrix plot."""
        import os
        result = subprocess.run(
            [sys.executable, "-m", "src.main"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert os.path.exists("reports/figures/confusion_matrix.png")
