"""Tests for UnivariateToMultivariate class."""
import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.univariate import UnivariateToMultivariate
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import ModelBase


class MockUnivariateModel(ModelBase):
    """Mock univariate model for testing."""

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
        if not isinstance(lookback_data, pd.DataFrame):
            raise ValueError("lookback_data must be DataFrame")
        return pd.DataFrame(
            np.ones((horizon, 1)),
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
        return np.ones((n_windows, horizon, 1))

    @property
    def model_name(self):
        return "MockUnivariateModel"


@pytest.fixture
def model_factory():
    """Create model factory for testing."""
    return lambda: MockUnivariateModel()


@pytest.fixture
def sample_data():
    """Create sample multivariate time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
        },
        index=dates,
    )


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
    }


@pytest.fixture
def sample_grouped_data():
    rows = []
    dates = pd.date_range(start="2023-01-01", periods=24, freq="D")
    for gid in ["S1", "S2", "S3"]:
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
def sample_grouped_covariates():
    rows = []
    static_rows = []
    dates = pd.date_range(start="2023-01-01", periods=24, freq="D")
    for gid_idx, gid in enumerate(["S1", "S2", "S3"]):
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


def test_init(model_factory):
    """Test initialization."""
    approach = UnivariateToMultivariate(model_factory)
    assert len(approach.models) == 0
    assert callable(approach.model_factory)


def test_forecast_fit(model_factory, sample_data, sample_covariates):
    """Test model fitting."""
    approach = UnivariateToMultivariate(model_factory)

    # Test basic fit
    approach.forecast_fit(sample_data)
    assert len(approach.models) == 3  # One model per column
    for col, model in approach.models.items():
        assert model.fitted
        assert model.train_data.shape[1] == 1
        assert model.train_data.columns[0] == col

    # Test fit with covariates
    approach = UnivariateToMultivariate(model_factory)
    approach.forecast_fit(sample_data, covariates=sample_covariates)
    assert len(approach.models) == 3
    for model in approach.models.values():
        assert model.fitted


def test_forecast(model_factory, sample_data, sample_covariates):
    """Test forecasting."""
    approach = UnivariateToMultivariate(model_factory)
    horizon = 5

    # Test forecast before fitting
    with pytest.raises(RuntimeError, match="Model not fitted"):
        approach.forecast(horizon, sample_data[[sample_data.columns[0]]])

    # Test basic forecast
    approach.forecast_fit(sample_data)
    result = approach.forecast(horizon, sample_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 3)  # 3 columns from sample_data
    assert isinstance(result.index, pd.DatetimeIndex)
    assert all(col in result.columns for col in sample_data.columns)

    # Test forecast with covariates
    result = approach.forecast(horizon, sample_data, covariates=sample_covariates)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 3)


def test_batch_forecast(model_factory, sample_data, sample_covariates):
    """Test batch forecasting."""
    approach = UnivariateToMultivariate(model_factory)
    horizon = 5

    # Create batch maker
    eval_maker = RollingForecastEvalBatchMaker(
        series=sample_data, index_list=[10, 20, 30], covariates=sample_covariates
    )
    batch_maker = RollingForecastPredictBatchMaker(eval_maker)

    # Test without fitting
    with pytest.raises(RuntimeError, match="Model not fitted"):
        approach.batch_forecast(horizon, batch_maker)

    # Test normal batch forecast
    approach.forecast_fit(sample_data)
    result = approach.batch_forecast(horizon, batch_maker)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 5, 3)  # 3 windows × horizon × n_cols

    # Test with invalid batch maker
    with pytest.raises(ValueError):
        approach.batch_forecast(horizon, None)


def test_process_covariates(model_factory, sample_covariates):
    """Test covariate processing."""
    approach = UnivariateToMultivariate(model_factory)

    # Test with valid covariates
    processed = approach._process_covariates(sample_covariates, col_idx=0)
    assert isinstance(processed, dict)
    assert "past_covariates" in processed
    assert "future_covariates" in processed
    assert all(df.shape[1] == 1 for df in processed.values())

    # Test with None covariates
    processed = approach._process_covariates(None, col_idx=0)
    assert isinstance(processed, dict)
    assert len(processed) == 0


def test_create_window_index(model_factory, sample_data):
    """Test window index creation."""
    approach = UnivariateToMultivariate(model_factory)

    # Test with DatetimeIndex
    idx = approach._create_window_index(sample_data.index, start=5, horizon=3)
    assert isinstance(idx, pd.DatetimeIndex)
    assert len(idx) == 3

    # Test with RangeIndex
    range_idx = pd.RangeIndex(0, 100)
    idx = approach._create_window_index(range_idx, start=5, horizon=3)
    assert isinstance(idx, pd.RangeIndex)
    assert len(idx) == 3


def test_grouped_forecast_fit_and_forecast(
    model_factory, sample_grouped_data, sample_grouped_covariates
):
    """Grouped mode must key models by group id and return long-format predictions."""
    approach = UnivariateToMultivariate(model_factory)
    grouped_kwargs = {
        "group_col": "series_id",
        "group_cols": ["series_id"],
        "time_col": "timestamp",
        "target_cols": ["y"],
    }

    approach.forecast_fit(
        sample_grouped_data,
        covariates=sample_grouped_covariates,
        **grouped_kwargs,
    )
    assert approach.grouped_mode is True
    assert len(approach.models) == 3
    assert ("S1", "y") in approach.models

    pred_df = approach.forecast(
        4,
        sample_grouped_data,
        covariates=sample_grouped_covariates,
        **grouped_kwargs,
    )
    assert isinstance(pred_df, pd.DataFrame)
    assert set(["timestamp", "series_id", "y"]).issubset(pred_df.columns)
    assert pred_df["series_id"].nunique() == 3
    assert len(pred_df) == 12
    assert pred_df["timestamp"].min() > sample_grouped_data["timestamp"].max()
