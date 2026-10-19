"""Tests for DefaultApproach class."""
import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.default import DefaultApproach
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import ModelBase


class MockModel(ModelBase):
    """Mock model for testing DefaultApproach."""

    def __init__(self, return_type="dataframe"):
        self.return_type = return_type
        self.fitted = False
        self.forecast_params = None
        self.train_data = None

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fitted = True
        self.train_data = train_data
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")

        self.forecast_params = {
            "horizon": horizon,
            "lookback_data": lookback_data,
            "covariates": covariates,
            "kwargs": kwargs,
        }

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

        n_cols = lookback_data.shape[1]
        if self.return_type == "dataframe":
            return pd.DataFrame(
                np.ones((horizon, n_cols)),
                columns=lookback_data.columns,
                index=pd.date_range(
                    start=lookback_data.index[-1],
                    periods=horizon + 1,
                    freq=pd.infer_freq(lookback_data.index),
                )[1:],
            )
        elif self.return_type == "array":
            return np.ones((horizon, n_cols))
        elif self.return_type == "series":
            # For series return type, convert to DataFrame with all columns
            series_data = pd.Series(np.ones(horizon), name=lookback_data.columns[0])
            return pd.DataFrame({col: series_data for col in lookback_data.columns})
        else:
            return "invalid"

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        if not isinstance(batch_maker, RollingForecastPredictBatchMaker):
            raise ValueError("Invalid batch maker type")
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        return np.ones(
            (
                len(batch_maker._batch_maker.index_list),
                horizon,
                self.train_data.shape[1],
            )
        )

    @property
    def model_name(self):
        return "MockModel"


@pytest.fixture
def model_factory():
    """Create model factory for testing."""
    return lambda: MockModel()


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {"A": np.random.randn(100), "B": np.random.randn(100)}, index=dates
    )


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
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
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


def test_init(model_factory):
    """Test initialization."""
    approach = DefaultApproach(model_factory)
    assert approach.model is None
    assert callable(approach.model_factory)


def test_forecast_fit(model_factory, sample_data, sample_covariates):
    """Test model fitting."""
    approach = DefaultApproach(model_factory)

    # Test basic fit
    approach.forecast_fit(sample_data)
    assert approach.model is not None
    assert approach.model.fitted

    # Test fit with covariates
    approach.forecast_fit(sample_data, covariates=sample_covariates)
    assert approach.model.fitted


def test_forecast(model_factory, sample_data, sample_covariates):
    """Test forecasting."""
    approach = DefaultApproach(model_factory)
    horizon = 5

    # Test forecast without fitting
    with pytest.raises(RuntimeError):
        approach.forecast(horizon, sample_data)

    # Test basic forecast
    approach.forecast_fit(sample_data)
    result = approach.forecast(horizon, sample_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 2)
    assert isinstance(result.index, pd.DatetimeIndex)

    # Test forecast with covariates
    result = approach.forecast(horizon, sample_data, covariates=sample_covariates)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (horizon, 2)

    # Test different return types
    for return_type in ["array", "series", "dataframe"]:
        approach = DefaultApproach(lambda: MockModel(return_type))
        approach.forecast_fit(sample_data)
        result = approach.forecast(horizon, sample_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (horizon, 2)

    # Test invalid return type
    approach = DefaultApproach(lambda: MockModel("invalid"))
    approach.forecast_fit(sample_data)
    with pytest.raises(TypeError):
        approach.forecast(horizon, sample_data)


def test_batch_forecast(model_factory, sample_data):
    """Test batch forecasting."""
    approach = DefaultApproach(model_factory)
    horizon = 5

    # Test without fitting
    eval_maker = RollingForecastEvalBatchMaker(
        series=sample_data, index_list=[10, 20, 30]
    )
    batch_maker = RollingForecastPredictBatchMaker(eval_maker)

    with pytest.raises(RuntimeError):
        approach.batch_forecast(horizon, batch_maker)

    # Test normal batch forecast
    approach.forecast_fit(sample_data)
    result = approach.batch_forecast(horizon, batch_maker)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 5, 2)  # 3 windows × horizon × n_features

    # Test with invalid batch maker
    with pytest.raises(ValueError):
        approach.batch_forecast(horizon, None)


def test_grouped_forecast_passthrough(model_factory, sample_grouped_data):
    """Grouped forecast must preserve long-format output from the model."""
    approach = DefaultApproach(model_factory)
    grouped_kwargs = {
        "group_col": "series_id",
        "group_cols": ["series_id"],
        "time_col": "timestamp",
        "target_cols": ["y"],
    }

    approach.forecast_fit(sample_grouped_data, **grouped_kwargs)
    result = approach.forecast(3, sample_grouped_data, **grouped_kwargs)

    assert isinstance(result, pd.DataFrame)
    assert set(["timestamp", "series_id", "y"]).issubset(result.columns)
    assert result["series_id"].nunique() == 3
    assert len(result) == 9
    assert approach.model.forecast_params["kwargs"]["group_col"] == "series_id"
