"""Tests for TimesFMAdapter — requires timesfm to be installed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_timesfm = pytest.importorskip(
    "timesfm",
    reason="timesfm not installed; skipping TimesFM adapter tests",
)
if not hasattr(_timesfm, "TimesFM_2p5_200M_torch"):
    pytest.skip(
        "timesfm<2.0 detected; need timesfm>=2.0.0 (TimesFM_2p5_200M_torch)",
        allow_module_level=True,
    )

from tsfb.base.forecasting.distribution_forecast import (  # noqa: E402,E501
    DistributionForecast,
)
from tsfb.base.models.foundation.adapters.timesfm_adapter import (  # noqa: E402,E501
    TimesFMAdapter,
)

_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


@pytest.fixture(scope="module")
def timesfm_adapter() -> TimesFMAdapter:
    """Single loaded adapter shared across module (model load is expensive)."""
    return TimesFMAdapter(
        pretrained_model_id=_MODEL_ID,
        device="cpu",
        inference_mode="zeroshot",
        torch_dtype="float32",
    )


@pytest.fixture
def series100() -> pd.DataFrame:
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="H")
    return pd.DataFrame({"value": np.cumsum(np.random.randn(100)) + 50.0}, index=idx)


class TestTimesFMAdapterInit:
    def test_model_name_format(self):
        a = TimesFMAdapter(pretrained_model_id=_MODEL_ID)
        assert a.model_name == f"timesfm[{_MODEL_ID}]"

    def test_capability_flags(self):
        a = TimesFMAdapter(pretrained_model_id=_MODEL_ID)
        assert a.supports_zeroshot is True
        assert a.supports_probabilistic is True
        assert a.supports_distribution is True
        assert a.output_type == "distribution"
        assert a.supports_finetuning is False  # removed in timesfm 2.x
        assert a.supports_fewshot is False

    def test_model_not_loaded_at_init(self):
        a = TimesFMAdapter(pretrained_model_id=_MODEL_ID)
        assert a._model is None

    def test_forecast_fit_noop_zeroshot(self, series100):
        a = TimesFMAdapter(pretrained_model_id=_MODEL_ID, inference_mode="zeroshot")
        result = a.forecast_fit(series100)
        assert result is a
        assert a._model is None

    def test_inference_mode_stored(self):
        for mode in ("zeroshot", "finetuned"):
            a = TimesFMAdapter(pretrained_model_id=_MODEL_ID, inference_mode=mode)
            assert a._inference_mode == mode

    def test_finetune_raises(self):
        a = TimesFMAdapter(pretrained_model_id=_MODEL_ID)
        with pytest.raises(NotImplementedError):
            a.finetune(None, None)  # type: ignore[arg-type]


@pytest.mark.slow
class TestTimesFMInference:
    def test_forecast_output_shape(self, timesfm_adapter, series100):
        preds = timesfm_adapter.forecast(12, series100)
        assert preds.shape == (12,)

    def test_forecast_dtype(self, timesfm_adapter, series100):
        preds = timesfm_adapter.forecast(12, series100)
        assert preds.dtype == np.float32

    def test_no_nan_in_output(self, timesfm_adapter, series100):
        preds = timesfm_adapter.forecast(12, series100)
        assert not np.any(np.isnan(preds))

    @pytest.mark.parametrize("horizon", [1, 6, 24])
    def test_various_horizons(self, timesfm_adapter, series100, horizon):
        preds = timesfm_adapter.forecast(horizon, series100)
        assert preds.shape == (horizon,)

    def test_model_loaded_after_forecast(self, timesfm_adapter, series100):
        timesfm_adapter.forecast(1, series100)
        assert timesfm_adapter._model is not None

    def test_raw_forecast_returns_tuple(self, timesfm_adapter, series100):
        mean_arr, quant_arr = timesfm_adapter._raw_forecast(series100)
        assert mean_arr.ndim == 1
        assert quant_arr.ndim == 2
        assert quant_arr.shape[1] == 9  # 9 native quantile levels

    def test_probabilistic_shape(self, timesfm_adapter, series100):
        samples = timesfm_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert samples.shape == (9, 12)

    def test_probabilistic_dtype(self, timesfm_adapter, series100):
        samples = timesfm_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert samples.dtype == np.float32

    def test_probabilistic_no_nan(self, timesfm_adapter, series100):
        samples = timesfm_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert not np.any(np.isnan(samples))

    def test_quantile_ordering(self, timesfm_adapter, series100):
        # Rows correspond to quantile levels 0.1..0.9 — first <= last
        samples = timesfm_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert np.all(samples[0] <= samples[-1] + 1e-5)

    def test_distribution_type(self, timesfm_adapter, series100):
        dist = timesfm_adapter.forecast_distribution(
            12, series100, distribution_name="normal"
        )
        assert isinstance(dist, DistributionForecast)
        assert dist.distribution_name == "normal"

    def test_distribution_params_shape(self, timesfm_adapter, series100):
        dist = timesfm_adapter.forecast_distribution(12, series100)
        assert dist.params["loc"].shape == (12,)
        assert dist.params["scale"].shape == (12,)

    def test_distribution_scale_positive(self, timesfm_adapter, series100):
        dist = timesfm_adapter.forecast_distribution(12, series100)
        assert np.all(dist.params["scale"] > 0)

    def test_batch_shape_and_dtype(self, timesfm_adapter, series100):
        from tsfb.base.evaluation.strategy.rolling_batch_maker import (  # noqa: PLC0415
            RollingForecastEvalBatchMaker,
            RollingForecastPredictBatchMaker,
        )

        predict_maker = RollingForecastPredictBatchMaker(
            RollingForecastEvalBatchMaker(series100, [50, 60, 70])
        )
        result = timesfm_adapter.batch_forecast(12, predict_maker)
        assert result.shape == (3, 12)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

    def test_batch_exhausts_maker(self, timesfm_adapter, series100):
        from tsfb.base.evaluation.strategy.rolling_batch_maker import (  # noqa: PLC0415
            RollingForecastEvalBatchMaker,
            RollingForecastPredictBatchMaker,
        )

        eval_maker = RollingForecastEvalBatchMaker(series100, [50, 60, 70])
        predict_maker = RollingForecastPredictBatchMaker(eval_maker)
        timesfm_adapter.batch_forecast(6, predict_maker)
        assert not predict_maker.has_more_batches()


class TestTimesFMSupportsBatch:
    def test_supports_batch_forecast_flag(self):
        assert (
            TimesFMAdapter(pretrained_model_id=_MODEL_ID).supports_batch_forecast
            is True
        )
