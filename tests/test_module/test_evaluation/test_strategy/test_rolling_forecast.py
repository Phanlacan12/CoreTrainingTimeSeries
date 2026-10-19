"""Tests for RollingForecast strategy."""
import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecast,
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import ModelBase


class MockModel(ModelBase):
    """Mock model for testing."""

    def __init__(self):
        self.fitted = False
        self.train_data = None
        self.config = {"output_chunk_shift": 0}

    @property
    def model_name(self) -> str:
        """Get model name."""
        return "mock_model"

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fitted = True
        self.train_data = train_data
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
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
        return np.ones((len(batch_maker._batch_maker.index_list), horizon, 1))


class MockApproach(ForecastingApproach):
    """Mock forecasting approach for testing."""

    def __init__(self):
        super().__init__()
        self.model = MockModel()

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.model.forecast_fit(train_data, covariates, **kwargs)
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        return self.model.forecast(horizon, lookback_data, covariates, **kwargs)

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        return self.model.batch_forecast(horizon, batch_maker, **kwargs)


class ShiftedRollingModel(MockModel):
    def __init__(self, output_shift: int):
        super().__init__()
        self.config = {"output_chunk_shift": output_shift}
        self.output_shift = output_shift

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        target_col = lookback_data.columns[0]
        last_value = float(lookback_data.iloc[-1, 0])
        freq = pd.infer_freq(lookback_data.index) or "D"
        return pd.DataFrame(
            {
                target_col: last_value
                + self.output_shift
                + np.arange(1, horizon + 1, dtype=float)
            },
            index=pd.date_range(
                start=lookback_data.index[-1],
                periods=horizon + self.output_shift + 1,
                freq=freq,
            )[self.output_shift + 1 :],
        )


class ShiftedRollingApproach(ForecastingApproach):
    def __init__(self, output_shift: int):
        super().__init__()
        self.model = ShiftedRollingModel(output_shift)

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.model.forecast_fit(train_data, covariates, **kwargs)
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        return self.model.forecast(horizon, lookback_data, covariates, **kwargs)

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        return self.model.batch_forecast(horizon, batch_maker, **kwargs)


class IdentityLoader:
    def inverse_predict(self, pred_df, series_name=None):
        return pred_df


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame(
        {"A": np.random.randn(100), "B": np.random.randn(100)}, index=dates
    )


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "strategy_name": "rolling_forecast",  # Add strategy name
        "horizon": 5,
        "stride": 2,
        "num_rollings": 3,
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
    }


@pytest.fixture
def evaluator():
    """Create evaluator for testing."""
    return Evaluator(["mae", "mse", "rmse"])


def test_rolling_forecast_init(sample_config, evaluator):
    """Test RollingForecast initialization."""
    strategy = RollingForecast(sample_config, evaluator)
    assert strategy.strategy_config == sample_config
    assert strategy.evaluator == evaluator


def test_get_index():
    """Test _get_index method."""
    indices = RollingForecast._get_index(
        train_length=50, test_length=20, horizon=5, stride=2
    )
    assert isinstance(indices, list)
    assert len(indices) > 0
    assert all(isinstance(i, int) for i in indices)
    assert indices[0] == 50  # Should start at train_length


def test_get_split_lens(sample_config, sample_data, evaluator):
    """Test _get_split_lens method."""
    strategy = RollingForecast(sample_config, evaluator)
    train_len, test_len = strategy._get_split_lens(sample_data, None, tv_ratio=0.8)
    assert train_len + test_len == len(sample_data)
    assert train_len > 0 and test_len > 0


def test_eval_sample(sample_config, sample_data, sample_covariates, evaluator):
    """Test _eval_sample method."""
    strategy = RollingForecast(sample_config, evaluator)
    approach = MockApproach()

    results = strategy._eval_sample(
        series=sample_data,
        meta_info=None,
        approach=approach,
        series_name="test_series",
        covariates=sample_covariates,
        loader=None,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert approach.model.fitted


def test_batch_maker():
    """Test RollingForecastEvalBatchMaker."""
    series = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    index_list = [3, 5, 7]
    batch_maker = RollingForecastEvalBatchMaker(series, index_list)

    # Test make_batch_predict
    batch = batch_maker.make_batch_predict(batch_size=2, win_size=2)
    assert "input" in batch
    assert "covariates" in batch
    assert "input_index" in batch

    # Test make_batch_eval
    eval_batch = batch_maker.make_batch_eval(horizon=2)
    assert "target" in eval_batch
    assert "covariates" in eval_batch


def test_rolling_forecast_with_errors(sample_config, sample_data, evaluator):
    """Test error handling in RollingForecast."""
    # Test with invalid split ratio
    invalid_config = sample_config.copy()
    invalid_config["data_module_config"]["split_ratio"]["train"] = 1.5
    strategy = RollingForecast(invalid_config, evaluator)

    with pytest.raises(ValueError):
        strategy._eval_sample(
            series=sample_data,
            meta_info=None,
            approach=MockApproach(),
            series_name="test_series",
            covariates=None,  # Add covariates argument
            loader=None,  # Add loader argument
        )

    # Test with missing required config
    incomplete_config = {k: v for k, v in sample_config.items() if k != "horizon"}
    with pytest.raises(RuntimeError, match="Missing options: horizon"):
        RollingForecast(incomplete_config, evaluator)

    # Test with missing another required config
    incomplete_config = {k: v for k, v in sample_config.items() if k != "strategy_name"}
    with pytest.raises(RuntimeError, match="Missing options: strategy_name"):
        RollingForecast(incomplete_config, evaluator)


def test_extract_covariates(sample_config, sample_covariates, evaluator):
    """Test covariate extraction methods."""
    strategy = RollingForecast(sample_config, evaluator)

    # Test training covariates extraction
    train_covs = strategy._extract_covariates(sample_covariates, train_len=50)
    assert isinstance(train_covs, dict)
    assert "past_covariates" in train_covs or "future_covariates" in train_covs

    # Test prediction covariates extraction
    pred_covs = strategy._extract_covariates_for_prediction(
        sample_covariates, start=50, horizon=5
    )
    assert isinstance(pred_covs, dict)
    assert "past_covariates" in pred_covs or "future_covariates" in pred_covs
    assert len(pred_covs["past_covariates"]) == 55

    shifted_pred_covs = strategy._extract_covariates_for_prediction(
        sample_covariates, start=50, horizon=5, output_shift=2
    )
    assert len(shifted_pred_covs["past_covariates"]) == 57


def test_rolling_forecast_single_series_respects_output_chunk_shift():
    strategy = RollingForecast(
        {
            "strategy_name": "rolling_forecast",
            "horizon": 3,
            "stride": 1,
            "num_rollings": 2,
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
        approach=ShiftedRollingApproach(output_shift=2),
        series_name="demo",
        covariates=None,
        loader=IdentityLoader(),
    )

    assert results[0] == pytest.approx(0.0, abs=1e-12)
