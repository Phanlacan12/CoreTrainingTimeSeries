"""Tests for metrics module."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsfb.base.evaluation.metrics.metrics import (
    MAE,
    MAPE,
    MSE,
    MSMAPE,
    RMSE,
    SMAPE,
    WAPE,
    BaseMetric,
    get_instantiated_metric_dict,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing metrics."""
    np.random.seed(42)
    true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Add some noise to create predictions
    pred_values = true_values + np.random.normal(0, 0.1, size=5)

    return pd.DataFrame({"true": true_values, "pred": pred_values})


@pytest.fixture
def sample_scaler():
    """Create a mock scaler for testing."""

    class MockScaler:
        def transform(self, data):
            if isinstance(data, pd.DataFrame):
                return data.values * 2.0
            return data * 2.0

    return MockScaler()


def test_base_metric():
    """Test BaseMetric class."""

    # Define TestMetric locally so it won't be picked up by get_instantiated_metric_dict
    class _TestMetric(BaseMetric):
        name = "test"

        def compute_scores(self, df, label_col, predicted_col, **kwargs):
            return 0.0

    metric = _TestMetric()
    assert metric.name == "test"
    assert callable(metric.compute_scores)


def test_mae_metric(sample_data):
    """Test MAE metric calculation."""
    mae = MAE()
    score = mae.compute_scores(sample_data, "true", "pred")
    expected_score = mean_absolute_error(sample_data["true"], sample_data["pred"])
    assert np.isclose(score, expected_score)


def test_mse_metric(sample_data):
    """Test MSE metric calculation."""
    mse = MSE()
    score = mse.compute_scores(sample_data, "true", "pred")
    expected_score = mean_squared_error(sample_data["true"], sample_data["pred"])
    assert np.isclose(score, expected_score)


def test_rmse_metric(sample_data):
    """Test RMSE metric calculation."""
    rmse = RMSE()
    score = rmse.compute_scores(sample_data, "true", "pred")
    expected_score = np.sqrt(
        mean_squared_error(sample_data["true"], sample_data["pred"])
    )
    assert np.isclose(score, expected_score)


def test_mape_metric(sample_data):
    """Test MAPE metric calculation."""
    mape = MAPE()
    score = mape.compute_scores(sample_data, "true", "pred")
    expected = (
        np.mean(
            np.abs((sample_data["true"] - sample_data["pred"]) / sample_data["true"])
        )
        * 100
    )
    assert np.isclose(score, expected)

    # Test with epsilon
    score_eps = mape.compute_scores(sample_data, "true", "pred", epsilon=1.0)
    assert not np.isnan(score_eps)


def test_smape_metric(sample_data):
    """Test SMAPE metric calculation."""
    smape = SMAPE()
    score = smape.compute_scores(sample_data, "true", "pred")
    numerator = np.abs(sample_data["true"] - sample_data["pred"])
    denominator = np.abs(sample_data["true"]) + np.abs(sample_data["pred"])
    expected = np.mean(2.0 * numerator / denominator) * 100
    assert np.isclose(score, expected)


def test_wape_metric(sample_data):
    """Test WAPE metric calculation."""
    wape = WAPE()
    score = wape.compute_scores(sample_data, "true", "pred")
    expected = (
        np.sum(np.abs(sample_data["true"] - sample_data["pred"]))
        / np.sum(np.abs(sample_data["true"]))
    ) * 100
    assert np.isclose(score, expected)


def test_msmape_metric(sample_data):
    """Test MSMAPE metric calculation."""
    msmape = MSMAPE()
    score = msmape.compute_scores(sample_data, "true", "pred")

    # Calculate expected score
    actual = sample_data["true"].values
    predicted = sample_data["pred"].values
    epsilon = 0.1
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    expected = np.mean(2 * np.abs(predicted - actual) / denom) * 100

    assert np.isclose(score, expected)

    # Test with custom epsilon
    score_eps = msmape.compute_scores(sample_data, "true", "pred", epsilon=0.5)
    assert not np.isnan(score_eps)


def test_get_instantiated_metric_dict():
    """Test getting dictionary of metric instances."""
    metrics = get_instantiated_metric_dict()
    assert isinstance(metrics, dict)
    assert all(isinstance(m, BaseMetric) for m in metrics.values())
    normalized_metric_keys = {name.lower() for name in metrics}
    expected_metrics = {
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "wape",
        "msmape",
        "r2",
        "r2_naive",
        "coverage",
        "joint_coverage",
        "mean_interval_width",
    }
    assert normalized_metric_keys == expected_metrics


def test_metrics_with_empty_data():
    """Test metrics with empty DataFrame."""
    empty_df = pd.DataFrame({"true": [], "pred": []})
    metrics = [MAE(), MSE(), RMSE(), MAPE(), SMAPE(), WAPE(), MSMAPE()]

    for metric in metrics:
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            metric.compute_scores(empty_df, "true", "pred")


def test_metrics_with_nan_values():
    """Test metrics with NaN values."""
    df_with_nan = pd.DataFrame({"true": [1.0, np.nan, 3.0], "pred": [1.1, 2.1, np.nan]})
    metrics = [MAE(), MSE(), RMSE(), MAPE(), SMAPE(), WAPE(), MSMAPE()]

    for metric in metrics:
        cleaned_df = metric._check_input(df_with_nan, "true", "pred")
        assert cleaned_df.shape[0] == 1  # Only one row should remain after cleaning


def test_metrics_with_mismatched_columns():
    """Test metrics with mismatched column names."""
    df = pd.DataFrame({"actual": [1, 2, 3], "forecast": [1.1, 2.1, 3.1]})
    metrics = [MAE(), MSE(), RMSE(), MAPE(), SMAPE(), WAPE(), MSMAPE()]

    for metric in metrics:
        with pytest.raises(KeyError):
            metric.compute_scores(df, "true", "pred")
