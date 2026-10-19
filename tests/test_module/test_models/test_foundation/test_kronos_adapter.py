"""Tests for KronosAdapter — requires vendor/Kronos to be installed via install.sh."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "kronos",
    reason="kronos not installed; run install.sh first to set up vendor/Kronos",
)

from tsfb.base.forecasting.distribution_forecast import (  # noqa: E402,E501
    DistributionForecast,
)
from tsfb.base.models.foundation.adapters.kronos_adapter import (  # noqa: E402,E501
    KronosAdapter,
)

_MODEL_ID = "NeoQuasar/Kronos-base"
_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"


@pytest.fixture(scope="module")
def kronos_adapter() -> KronosAdapter:
    """Single loaded adapter shared across module (model load is expensive)."""
    return KronosAdapter(
        pretrained_model_id=_MODEL_ID,
        tokenizer_id=_TOKENIZER_ID,
        device="cpu",
        inference_mode="zeroshot",
        torch_dtype="float32",
    )


@pytest.fixture
def series_with_datetimeindex() -> pd.DataFrame:
    """Kronos requires a DatetimeIndex."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100.0}, index=idx)


class TestKronosAdapterInit:
    def test_model_name_format(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        assert a.model_name == f"kronos[{_MODEL_ID}]"

    def test_capability_flags(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        assert a.supports_zeroshot is True
        assert a.supports_probabilistic is True
        assert a.supports_finetuning is False  # KronosPredictor has no fine-tune API
        assert a.supports_fewshot is False

    def test_predictor_not_loaded_at_init(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        assert a._predictor is None

    def test_temperature_stored(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID, temperature=0.9)
        assert a._temperature == 0.9

    def test_tokenizer_id_stored(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID, tokenizer_id=_TOKENIZER_ID)
        assert a._tokenizer_id == _TOKENIZER_ID

    def test_forecast_fit_noop_zeroshot(self, series_with_datetimeindex):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID, inference_mode="zeroshot")
        result = a.forecast_fit(series_with_datetimeindex)
        assert result is a
        assert a._predictor is None


class TestKronosHelpers:
    def test_series_to_ohlcv_shape(self, series_with_datetimeindex):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        ohlcv_df, x_ts = a._series_to_ohlcv(series_with_datetimeindex)
        expected_cols = ["open", "high", "low", "close", "volume", "amount"]
        assert list(ohlcv_df.columns) == expected_cols
        assert len(ohlcv_df) == len(series_with_datetimeindex)

    def test_series_to_ohlcv_ohlc_equal_close(self, series_with_datetimeindex):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        ohlcv_df, _ = a._series_to_ohlcv(series_with_datetimeindex)
        # open/high/low should all equal close for univariate input
        for col in ("open", "high", "low"):
            np.testing.assert_array_equal(
                ohlcv_df[col].values, ohlcv_df["close"].values
            )

    def test_series_to_ohlcv_context_length_limit(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID, context_length=50)
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        series = pd.DataFrame({"v": np.ones(200)}, index=idx)
        ohlcv_df, _ = a._series_to_ohlcv(series)
        assert len(ohlcv_df) == 50

    def test_series_to_ohlcv_requires_datetimeindex(self):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        series = pd.DataFrame({"v": [1.0, 2.0, 3.0]})  # no DatetimeIndex
        with pytest.raises(ValueError, match="DatetimeIndex"):
            a._series_to_ohlcv(series)

    def test_future_index_length(self, series_with_datetimeindex):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        idx = a._future_index(series_with_datetimeindex, horizon=12)
        assert len(idx) == 12

    def test_future_index_starts_after_series(self, series_with_datetimeindex):
        a = KronosAdapter(pretrained_model_id=_MODEL_ID)
        idx = a._future_index(series_with_datetimeindex, horizon=5)
        assert idx[0] > series_with_datetimeindex.index[-1]


@pytest.mark.slow
class TestKronosInference:
    def test_forecast_output_shape(self, kronos_adapter, series_with_datetimeindex):
        preds = kronos_adapter.forecast(12, series_with_datetimeindex)
        assert preds.shape == (12,)

    def test_forecast_dtype(self, kronos_adapter, series_with_datetimeindex):
        preds = kronos_adapter.forecast(12, series_with_datetimeindex)
        assert preds.dtype == np.float32

    def test_no_nan_in_output(self, kronos_adapter, series_with_datetimeindex):
        preds = kronos_adapter.forecast(12, series_with_datetimeindex)
        assert not np.any(np.isnan(preds))

    @pytest.mark.parametrize("horizon", [1, 6, 24])
    def test_various_horizons(self, kronos_adapter, series_with_datetimeindex, horizon):
        preds = kronos_adapter.forecast(horizon, series_with_datetimeindex)
        assert preds.shape == (horizon,)

    def test_predictor_loaded_after_forecast(
        self, kronos_adapter, series_with_datetimeindex
    ):
        kronos_adapter.forecast(1, series_with_datetimeindex)
        assert kronos_adapter._predictor is not None

    def test_probabilistic_shape(self, kronos_adapter, series_with_datetimeindex):
        samples = kronos_adapter.forecast_probabilistic(
            12, series_with_datetimeindex, num_samples=10
        )
        assert samples.shape == (10, 12)

    def test_probabilistic_dtype(self, kronos_adapter, series_with_datetimeindex):
        samples = kronos_adapter.forecast_probabilistic(
            12, series_with_datetimeindex, num_samples=10
        )
        assert samples.dtype == np.float32

    def test_probabilistic_no_nan(self, kronos_adapter, series_with_datetimeindex):
        samples = kronos_adapter.forecast_probabilistic(
            12, series_with_datetimeindex, num_samples=10
        )
        assert not np.any(np.isnan(samples))

    def test_temperature_variation_produces_diversity(
        self, kronos_adapter, series_with_datetimeindex
    ):
        # Temperature-varied sampling should produce non-identical trajectories
        samples = kronos_adapter.forecast_probabilistic(
            12, series_with_datetimeindex, num_samples=20
        )
        row_stds = samples.std(axis=0)
        assert np.any(
            row_stds > 0
        ), "Temperature sampling produced identical trajectories"  # noqa: E501

    def test_distribution_type(self, kronos_adapter, series_with_datetimeindex):
        dist = kronos_adapter.forecast_distribution(
            12, series_with_datetimeindex, distribution_name="normal"
        )
        assert isinstance(dist, DistributionForecast)
        assert dist.distribution_name == "normal"

    def test_distribution_params_shape(self, kronos_adapter, series_with_datetimeindex):
        dist = kronos_adapter.forecast_distribution(12, series_with_datetimeindex)
        assert dist.params["loc"].shape == (12,)
        assert dist.params["scale"].shape == (12,)
