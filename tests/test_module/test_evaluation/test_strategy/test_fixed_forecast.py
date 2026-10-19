"""Tests for FixedForecast strategy."""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy.fixed_forecast import (
    CovariateData,
    FixedForecast,
    ForecastConfig,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self.fitted = False
        self.config = {"output_chunk_shift": 0}

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fitted = True
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        return pd.DataFrame(
            np.ones((horizon, lookback_data.shape[1])),
            columns=lookback_data.columns,
            index=pd.date_range(
                start=lookback_data.index[-1],
                periods=horizon + 1,
                freq=pd.infer_freq(lookback_data.index),
            )[1:],
        )


@pytest.fixture
def evaluator():
    """Create evaluator for testing."""
    return Evaluator(["mae", "mse", "rmse"])


class MockApproach(ForecastingApproach):
    def __init__(self):
        super().__init__()
        self.model = MockModel()

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.model.forecast_fit(train_data, covariates, **kwargs)
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        return self.model.forecast(horizon, lookback_data, covariates, **kwargs)

    def batch_forecast(self, *args, **kwargs):
        raise NotImplementedError("Not needed for this test.")


class ShiftedSingleModel(MockModel):
    def __init__(self, output_shift: int):
        super().__init__()
        self.config = {"output_chunk_shift": output_shift}
        self.output_shift = output_shift

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        last_value = float(lookback_data.iloc[-1, 0])
        freq = pd.infer_freq(lookback_data.index) or "D"
        return pd.DataFrame(
            {
                lookback_data.columns[0]: last_value
                + self.output_shift
                + np.arange(1, horizon + 1, dtype=float)
            },
            index=pd.date_range(
                start=lookback_data.index[-1],
                periods=horizon + self.output_shift + 1,
                freq=freq,
            )[self.output_shift + 1 :],
        )


class ShiftedSingleApproach(ForecastingApproach):
    def __init__(self, output_shift: int):
        super().__init__()
        self.model = ShiftedSingleModel(output_shift)

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.model.forecast_fit(train_data, covariates, **kwargs)
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        return self.model.forecast(horizon, lookback_data, covariates, **kwargs)

    def batch_forecast(self, *args, **kwargs):
        raise NotImplementedError("Not needed for this test.")


class MockLoader:
    """Mock data loader for testing."""

    def __init__(self, with_scaler=True):
        if with_scaler:
            self.target_scalers = StandardScaler()
            # Fit scaler with dummy data
            dummy_data = pd.DataFrame(np.random.randn(10, 2), columns=["A", "B"])
            self.target_scalers.fit(dummy_data)
        else:
            self.target_scalers = None


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {"A": np.random.randn(100), "B": np.random.randn(100)}, index=dates
    )


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "strategy_name": "fixed_forecast",
        "horizon": 5,
        "save_true_pred": True,
        "target_channel": None,
        "seed": 42,
        "deterministic": "full",
        "data_module_config": {"split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2}},
    }


@pytest.fixture
def sample_covariates(sample_data):
    """Create sample covariates."""
    return {
        "past_covariates": pd.DataFrame(
            np.random.randn(100, 2), index=sample_data.index, columns=["cov1", "cov2"]
        ),
        "future_covariates": pd.DataFrame(
            np.random.randn(100, 2), index=sample_data.index, columns=["fcov1", "fcov2"]
        ),
        "static_covariates": pd.DataFrame(
            np.random.randn(100, 1), index=sample_data.index, columns=["static1"]
        ),
    }


def test_forecast_config():
    """Test ForecastConfig dataclass."""
    config = ForecastConfig(
        horizon=5,
        save_tp=True,
        train_ratio=0.8,
        train_len=80,
        calib_len=0,
        prediction_start=80,
        output_shift=0,
        is_multi=True,
    )
    assert config.horizon == 5
    assert config.save_tp is True
    assert config.train_ratio == 0.8
    assert config.train_len == 80
    assert config.prediction_start == 80
    assert config.is_multi is True


def test_covariate_data(sample_covariates):
    """Test CovariateData class."""
    covariate_data = CovariateData.from_raw_data(
        sample_covariates, train_len=80, horizon=5
    )

    assert isinstance(covariate_data.train, dict)
    assert isinstance(covariate_data.pred, dict)
    assert "past_covariates" in covariate_data.train
    assert "future_covariates" in covariate_data.train
    assert "static_covariates" in covariate_data.train
    assert len(covariate_data.pred["past_covariates"]) == 85

    shifted_data = CovariateData.from_raw_data(
        sample_covariates, train_len=80, horizon=5, output_shift=2
    )
    assert len(shifted_data.pred["past_covariates"]) == 87

    # Test with None covariates
    empty_data = CovariateData.from_raw_data(None, train_len=80, horizon=5)
    assert len(empty_data.train) == 0
    assert len(empty_data.pred) == 0


def test_fixed_forecast_init(sample_config, evaluator):
    """Test FixedForecast initialization."""
    strategy = FixedForecast(sample_config, evaluator)
    assert strategy.strategy_config == sample_config
    assert strategy.evaluator == evaluator


def test_default_interval_payload_dir_uses_session_artifact_root(
    sample_config, evaluator
):
    """Default interval payloads should stay inside the shared test artifact root."""
    strategy = FixedForecast(sample_config, evaluator)
    resolved_dir = strategy._resolve_payload_output_dir()

    artifact_root = os.environ["TSFB_ARTIFACT_ROOT"]
    assert resolved_dir == os.path.join(artifact_root, "interval_payloads")


def test_get_forecast_config(sample_config, sample_data, evaluator):
    """Test _get_forecast_config method."""
    strategy = FixedForecast(sample_config, evaluator)
    config = strategy._get_forecast_config(sample_data, "test_series", 0)

    assert isinstance(config, ForecastConfig)
    assert config.horizon == sample_config["horizon"]
    assert config.save_tp == sample_config["save_true_pred"]
    assert 0 < config.train_ratio < 1
    assert config.train_len > 0


def test_normalize_data(sample_data, evaluator, sample_config):
    strategy = FixedForecast(sample_config, evaluator)

    # Test with scaler
    loader_with_scaler = MockLoader(with_scaler=True)
    norm_data = strategy._normalize_data(sample_data, loader_with_scaler)
    assert isinstance(norm_data, pd.DataFrame)
    assert norm_data.shape == sample_data.shape

    # Test without scaler
    loader_without_scaler = MockLoader(with_scaler=False)
    unchanged_data = strategy._normalize_data(sample_data, loader_without_scaler)
    pd.testing.assert_frame_equal(unchanged_data, sample_data)


def test_fit_and_predict(sample_config, sample_data, sample_covariates, evaluator):
    """Test _fit_and_predict method."""
    strategy = FixedForecast(sample_config, evaluator)
    approach = MockApproach()
    config = strategy._get_forecast_config(sample_data, "test_series", 0)
    covariate_data = CovariateData.from_raw_data(
        sample_covariates, config.train_len, config.horizon
    )

    norm_train_df = sample_data.iloc[: config.train_len]
    pred_df, fit_time, predict_time = strategy._fit_and_predict(
        approach, config, norm_train_df, covariate_data
    )

    assert isinstance(pred_df, pd.DataFrame)
    assert isinstance(fit_time, float)
    assert isinstance(predict_time, float)
    assert approach.model.fitted


def test_prepare_prediction(sample_config, sample_data, evaluator):
    """Test _prepare_prediction method."""
    strategy = FixedForecast(sample_config, evaluator)
    config = strategy._get_forecast_config(sample_data, "test_series", 0)

    # Test with DataFrame prediction
    pred_df = pd.DataFrame(
        np.ones((5, 2)),
        columns=sample_data.columns,
        index=pd.date_range("2023-01-01", periods=5),
    )
    test_df = sample_data.iloc[-5:]

    y_true, y_pred, pred_df_inv, target_cols = strategy._prepare_prediction(
        pred_df, test_df, config, "test_series", None
    )

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(pred_df_inv, pd.DataFrame)
    assert isinstance(target_cols, list)
    assert y_true.shape == y_pred.shape


def test_execute(sample_config, sample_data, sample_covariates, evaluator):
    """Test full execution of FixedForecast strategy."""
    strategy = FixedForecast(sample_config, evaluator)
    approach = MockApproach()

    results = strategy._execute(
        series=sample_data,
        meta_info=None,
        approach=approach,
        series_name="test_series",
        covariates=sample_covariates,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert approach.model.fitted


def test_fixed_forecast_single_series_respects_output_chunk_shift(evaluator):
    strategy = FixedForecast(
        {
            "strategy_name": "fixed_forecast",
            "horizon": 3,
            "save_true_pred": False,
            "target_channel": "target",
            "seed": 42,
            "deterministic": "full",
            "data_module_config": {
                "split_ratio": {"train": 0.5, "val": 0.2, "test": 0.3}
            },
        },
        Evaluator(["mae"]),
    )
    series = pd.DataFrame(
        {"target": np.arange(20, dtype=float)},
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )

    results = strategy._execute(
        series=series,
        meta_info=None,
        approach=ShiftedSingleApproach(output_shift=2),
        series_name="demo",
        covariates=None,
        loader=None,
    )

    assert results[0] == pytest.approx(0.0, abs=1e-12)


def test_fixed_forecast_with_errors(sample_config, sample_data, evaluator):
    """Test error handling in FixedForecast."""
    # Case 1: Invalid train + val > 1.0
    invalid_config = sample_config.copy()
    invalid_config["data_module_config"]["split_ratio"]["train"] = 0.9
    invalid_config["data_module_config"]["split_ratio"]["val"] = 0.2
    strategy = FixedForecast(invalid_config, evaluator)
    with pytest.raises(ValueError, match="Invalid train\\+val ratio"):
        strategy._get_forecast_config(sample_data, "test_series", 0)

    # Case 2: Missing required config
    incomplete_config = {k: v for k, v in sample_config.items() if k != "horizon"}
    with pytest.raises(RuntimeError):
        FixedForecast(incomplete_config, evaluator)

    # Case 3: Horizon > data length (ensure valid split ratio)
    too_large_horizon_config = sample_config.copy()
    too_large_horizon_config["horizon"] = 1000
    too_large_horizon_config["data_module_config"]["split_ratio"] = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    }
    strategy = FixedForecast(too_large_horizon_config, evaluator)
    with pytest.raises(ValueError, match="Prediction length exceeds data length"):
        strategy._get_forecast_config(sample_data, "test_series", 0)
