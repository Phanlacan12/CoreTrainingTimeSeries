import numpy as np
import pytest

from tsfb.base.evaluation.strategy.metric_utils import evaluate_metrics_with_group


class TestFairnessUtils:
    def test_evaluate_metrics_with_group_basic(self):
        """Test basic functionality of evaluate_metrics_with_group."""
        # Setup data
        # Group A: [1, 2], Pred: [1.1, 2.1] -> Error: [0.1, 0.1]. MAE=0.1, MSE=0.01
        # Group B: [3, 4], Pred: [3.2, 4.2] -> Error: [0.2, 0.2]. MAE=0.2, MSE=0.04
        # Global: MAE=(0.1*2 + 0.2*2)/4 = 0.15. MSE=(0.01*2 + 0.04*2)/4 = 0.025

        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 2.1, 3.2, 4.2])
        groups = ["A", "A", "B", "B"]
        metrics = ["mae", "mse"]

        results = evaluate_metrics_with_group(
            y_true=y_true, y_pred=y_pred, metric_names=metrics, groups=groups
        )

        # Verify Global
        # allow small float diff
        assert results["global"][0] == pytest.approx(0.15, abs=1e-5)
        assert results["global"][1] == pytest.approx(0.025, abs=1e-5)

        # Verify Per-Series
        per_series = results["per_series"]
        assert "A" in per_series
        assert "B" in per_series

        # Group A
        assert per_series["A"][0] == pytest.approx(0.1, abs=1e-5)
        assert per_series["A"][1] == pytest.approx(0.01, abs=1e-5)

        # Group B
        assert per_series["B"][0] == pytest.approx(0.2, abs=1e-5)
        assert per_series["B"][1] == pytest.approx(0.04, abs=1e-5)

    def test_evaluate_metrics_with_no_groups(self):
        """Test behavior when no groups are provided."""
        y_true = np.array([1, 2])
        y_pred = np.array([1.1, 2.1])
        metrics = ["mae"]

        results = evaluate_metrics_with_group(
            y_true=y_true, y_pred=y_pred, metric_names=metrics, groups=None
        )

        assert results["global"][0] == pytest.approx(0.1, abs=1e-5)
        assert results["per_series"] == {}
