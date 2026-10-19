"""Tests for HybridApproach class."""
import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.hybrid import HybridApproach
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import ModelBase


class MockForecastModel(ModelBase):
    """Mock model for testing."""

    def __init__(self):
        self.fitted = False
        self.train_data = None

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fitted = True
        self.train_data = train_data
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        group_col = kwargs.get("group_col")
        time_col = kwargs.get("time_col")
        target_cols = kwargs.get("target_cols") or list(lookback_data.columns)
        if (
            group_col is not None
            and time_col is not None
            and group_col in lookback_data.columns
            and time_col in lookback_data.columns
        ):
            outputs = []
            for gid, group_df in lookback_data.groupby(group_col, sort=True):
                group_df = group_df.sort_values(time_col).reset_index(drop=True)
                freq = pd.infer_freq(pd.to_datetime(group_df[time_col])) or "D"
                pred_idx = pd.date_range(
                    start=pd.to_datetime(group_df[time_col]).iloc[-1],
                    periods=horizon + 1,
                    freq=freq,
                )[1:]
                pred_df = pd.DataFrame({time_col: pred_idx, group_col: gid})
                for target_col in target_cols:
                    pred_df[target_col] = 1.0
                outputs.append(pred_df)
            return pd.concat(outputs, ignore_index=True)
        return pd.DataFrame(
            np.ones((horizon, lookback_data.shape[1])),
            columns=lookback_data.columns,
            index=pd.date_range(
                start=lookback_data.index[-1],
                periods=horizon + 1,
                freq=pd.infer_freq(lookback_data.index),
            )[1:],
        )

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        if not isinstance(batch_maker, RollingForecastPredictBatchMaker):
            raise ValueError("Invalid batch maker type")
        n_windows = len(batch_maker._batch_maker.index_list)
        return np.ones((n_windows, horizon, self.train_data.shape[1]))

    @property
    def model_name(self):
        return "MockForecastModel"


@pytest.fixture
def sample_data():
    """Create sample multivariate time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
            "D": np.random.randn(100),
        },
        index=dates,
    )


@pytest.fixture
def sample_config():
    """Create sample hybrid approach configuration."""
    return {
        "group": [
            {
                "series": ["A", "B"],
                "sub_approach": "univariate_per_series",
                "model_name": "mock_model",
                "model_hyper_params": {},
            },
            {
                "series": ["C", "D"],
                "sub_approach": "multivariate",
                "model_name": "mock_model",
                "model_hyper_params": {},
            },
        ]
    }


@pytest.fixture
def sample_covariates(sample_data):
    """Create sample covariates."""
    return {
        "past_covariates": pd.DataFrame(
            np.random.randn(100, 2), index=sample_data.index, columns=["cov1", "cov2"]
        )
    }


@pytest.fixture
def sample_grouped_data():
    rows = []
    dates = pd.date_range(start="2023-01-01", periods=24, freq="D")
    for gid in ["S1", "S2", "S3", "S4"]:
        for idx, ts in enumerate(dates):
            rows.append(
                {
                    "timestamp": ts,
                    "series_id": gid,
                    "y": float(idx + 1),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_grouped_config():
    return {
        "group": [
            {
                "group_values": ["S1", "S2"],
                "sub_approach": "univariate_per_series",
                "model_name": "mock_model",
                "model_hyper_params": {},
            },
            {
                "group_values": ["S3", "S4"],
                "sub_approach": "multivariate",
                "model_name": "mock_model",
                "model_hyper_params": {},
            },
        ]
    }


@pytest.fixture
def sample_grouped_covariates():
    rows = []
    static_rows = []
    dates = pd.date_range(start="2023-01-01", periods=24, freq="D")
    for gid_idx, gid in enumerate(["S1", "S2", "S3", "S4"]):
        static_rows.append({"series_id": gid, "static1": float(gid_idx + 1)})
        for idx, ts in enumerate(dates):
            rows.append(
                {
                    "timestamp": ts,
                    "series_id": gid,
                    "past_cov": float(idx),
                    "future_cov": float(idx + 10),
                }
            )
    df = pd.DataFrame(rows)
    return {
        "past_covariates": df[["timestamp", "series_id", "past_cov"]].copy(),
        "future_covariates": df[["timestamp", "series_id", "future_cov"]].copy(),
        "static_covariates": pd.DataFrame(static_rows),
    }


def test_init(sample_config):
    """Test initialization."""
    approach = HybridApproach(sample_config)
    assert approach.config == sample_config
    assert len(approach.feature_names) == 0
    assert len(approach.models) == 0


def test_forecast_fit(monkeypatch, sample_data, sample_config, sample_covariates):
    """Test model fitting."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    approach = HybridApproach(sample_config)

    # Test basic fit
    approach.forecast_fit(sample_data)
    assert len(approach.models) == 2  # One for each group
    assert "A,B" in approach.models
    assert "C,D" in approach.models

    for group_key, (sub_approach, cols) in approach.models.items():
        if group_key == "A,B":
            assert len(sub_approach.models) == 2  # UnivariateToMultivariate
            for model in sub_approach.models.values():
                assert model.fitted
        else:
            assert sub_approach.model is not None  # DefaultApproach
            assert sub_approach.model.fitted

    # Test fit with covariates
    approach = HybridApproach(sample_config)
    approach.forecast_fit(sample_data, covariates=sample_covariates)
    assert len(approach.models) == 2


def test_forecast(monkeypatch, sample_data, sample_config, sample_covariates):
    """Test forecasting."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    approach = HybridApproach(sample_config)
    horizon = 5

    # Test forecast before fitting
    with pytest.raises(RuntimeError):
        approach.forecast(horizon, sample_data)

    # Test basic forecast
    approach.forecast_fit(sample_data)
    result = approach.forecast(horizon, sample_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 4)  # All columns from both groups
    assert all(col in result.columns for col in sample_data.columns)
    assert isinstance(result.index, pd.DatetimeIndex)

    # Test forecast with covariates
    result = approach.forecast(horizon, sample_data, covariates=sample_covariates)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 4)


def test_batch_forecast(monkeypatch, sample_data, sample_config, sample_covariates):
    """Test batch forecasting."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    approach = HybridApproach(sample_config)
    horizon = 5

    # Create batch maker
    eval_maker = RollingForecastEvalBatchMaker(
        series=sample_data, index_list=[10, 20, 30], covariates=sample_covariates
    )
    batch_maker = RollingForecastPredictBatchMaker(eval_maker)

    # Test without fitting
    with pytest.raises(RuntimeError):
        approach.batch_forecast(horizon, batch_maker)

    # Test normal batch forecast
    approach.forecast_fit(sample_data)
    result = approach.batch_forecast(horizon, batch_maker)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 5, 4)  # 3 windows × horizon × total_features

    # Test with invalid batch maker
    with pytest.raises(ValueError):
        approach.batch_forecast(horizon, None)


def test_invalid_config():
    """Test configuration validation."""
    # Test empty config
    empty_config = {}
    with pytest.raises(KeyError, match="group"):
        HybridApproach(empty_config).forecast_fit(pd.DataFrame())

    # Test invalid sub_approach
    config = {
        "group": [
            {"series": ["A"], "sub_approach": "invalid", "model_name": "mock_model"}
        ]
    }
    with pytest.raises(ValueError, match="Unknown sub_approach"):
        HybridApproach(config).forecast_fit(pd.DataFrame({"A": [1, 2, 3]}))


def test_grouped_hybrid_fit_and_forecast(
    monkeypatch,
    sample_grouped_data,
    sample_grouped_config,
    sample_grouped_covariates,
):
    """Grouped hybrid must route by group_values and return long-format predictions."""

    def mock_get_model(*args, **kwargs):
        return MockForecastModel()

    monkeypatch.setattr("tsfb.base.approaches.hybrid.get_single_model", mock_get_model)

    grouped_kwargs = {
        "group_col": "series_id",
        "group_cols": ["series_id"],
        "time_col": "timestamp",
        "target_cols": ["y"],
    }

    approach = HybridApproach(sample_grouped_config)
    approach.forecast_fit(
        sample_grouped_data,
        covariates=sample_grouped_covariates,
        **grouped_kwargs,
    )
    assert approach.grouped_mode is True
    assert len(approach.models) == 2

    pred_df = approach.forecast(
        3,
        sample_grouped_data,
        covariates=sample_grouped_covariates,
        **grouped_kwargs,
    )
    assert isinstance(pred_df, pd.DataFrame)
    assert set(["timestamp", "series_id", "y"]).issubset(pred_df.columns)
    assert pred_df["series_id"].nunique() == 4
    assert len(pred_df) == 12
