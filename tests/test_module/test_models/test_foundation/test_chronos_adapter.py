"""Tests for ChronosAdapter — requires chronos-forecasting to be installed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

chronos = pytest.importorskip(
    "chronos",
    reason="chronos-forecasting not installed; skipping Chronos adapter tests",
)

from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter
from tsfb.base.schema.foundation_config import FineTuneConfig

# Use the smallest available Chronos-2 model to keep tests fast
_MODEL_ID = "autogluon/chronos-2-small"


@pytest.fixture(scope="module")
def chronos_zeroshot() -> ChronosAdapter:
    """Single loaded adapter shared across module tests (model load is expensive)."""
    return ChronosAdapter(
        pretrained_model_id=_MODEL_ID,
        device="cpu",
        inference_mode="zeroshot",
        num_samples=9,
        torch_dtype="float32",
    )


@pytest.fixture
def series100() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({"value": np.cumsum(np.random.randn(100)) + 50.0})


class TestChronosAdapterInit:
    def test_model_name_format(self):
        adapter = ChronosAdapter(pretrained_model_id=_MODEL_ID)
        assert adapter.model_name == f"chronos[{_MODEL_ID}]"

    def test_capability_flags(self):
        adapter = ChronosAdapter(pretrained_model_id=_MODEL_ID)
        assert adapter.supports_zeroshot is True
        assert adapter.supports_finetuning is True
        assert adapter.supports_probabilistic is True
        assert adapter.supports_fewshot is False

    def test_pipeline_not_loaded_at_init(self):
        adapter = ChronosAdapter(pretrained_model_id=_MODEL_ID)
        assert adapter._pipeline is None

    def test_inference_mode_stored(self):
        for mode in ("zeroshot", "finetuned", "fewshot"):
            a = ChronosAdapter(pretrained_model_id=_MODEL_ID, inference_mode=mode)
            assert a._inference_mode == mode

    def test_finetune_config_stored_as_raw_dict(self):
        adapter = ChronosAdapter(
            pretrained_model_id=_MODEL_ID,
            finetune_config={"mode": "lora", "epochs": 3},
        )
        assert adapter._raw_finetune_config == {"mode": "lora", "epochs": 3}


class TestChronosForecastFitZeroshot:
    def test_forecast_fit_noop_returns_self(self, series100):
        adapter = ChronosAdapter(
            pretrained_model_id=_MODEL_ID,
            inference_mode="zeroshot",
        )
        result = adapter.forecast_fit(series100)
        assert result is adapter

    def test_pipeline_not_loaded_after_zeroshot_fit(self, series100):
        adapter = ChronosAdapter(
            pretrained_model_id=_MODEL_ID,
            inference_mode="zeroshot",
        )
        adapter.forecast_fit(series100)
        assert adapter._pipeline is None


class TestChronosZeroshotForecast:
    def test_output_shape(self, chronos_zeroshot, series100):
        horizon = 12
        preds = chronos_zeroshot.forecast(horizon, series100)
        assert preds.shape == (horizon,)

    def test_output_dtype(self, chronos_zeroshot, series100):
        preds = chronos_zeroshot.forecast(12, series100)
        assert preds.dtype == np.float32

    def test_no_nan_in_output(self, chronos_zeroshot, series100):
        preds = chronos_zeroshot.forecast(12, series100)
        assert not np.any(np.isnan(preds))

    @pytest.mark.parametrize("horizon", [1, 6, 24])
    def test_various_horizons(self, chronos_zeroshot, series100, horizon):
        preds = chronos_zeroshot.forecast(horizon, series100)
        assert preds.shape == (horizon,)

    def test_pipeline_loaded_after_forecast(self, chronos_zeroshot, series100):
        chronos_zeroshot.forecast(1, series100)
        assert chronos_zeroshot._pipeline is not None

    def test_short_context(self, chronos_zeroshot):
        short = pd.DataFrame({"v": np.arange(10, dtype=float)})
        preds = chronos_zeroshot.forecast(6, short)
        assert preds.shape == (6,)


class TestChronosProbabilistic:
    def test_output_shape(self, chronos_zeroshot, series100):
        num_samples = 9
        samples = chronos_zeroshot.forecast_probabilistic(
            12, series100, num_samples=num_samples
        )
        assert samples.shape == (num_samples, 12)

    def test_output_dtype(self, chronos_zeroshot, series100):
        samples = chronos_zeroshot.forecast_probabilistic(12, series100, num_samples=9)
        assert samples.dtype == np.float32

    def test_no_nan(self, chronos_zeroshot, series100):
        samples = chronos_zeroshot.forecast_probabilistic(12, series100, num_samples=9)
        assert not np.any(np.isnan(samples))

    def test_quantile_ordering(self, chronos_zeroshot, series100):
        samples = chronos_zeroshot.forecast_probabilistic(12, series100, num_samples=9)
        # Rows are quantile levels (0.1 .. 0.9); first row should be <= last row
        assert np.all(samples[0] <= samples[-1] + 1e-5)

    @pytest.mark.parametrize("num_samples", [1, 9, 18])
    def test_num_samples_parameter(self, chronos_zeroshot, series100, num_samples):
        samples = chronos_zeroshot.forecast_probabilistic(
            12, series100, num_samples=num_samples
        )
        assert samples.shape[0] == num_samples


class TestChronosBuildContext:
    def test_univariate_returns_1d(self):
        adapter = ChronosAdapter(pretrained_model_id=_MODEL_ID)
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        ctx = adapter._build_context(df)
        assert ctx.ndim == 1
        assert len(ctx) == 3

    def test_multivariate_takes_first_column(self):
        adapter = ChronosAdapter(pretrained_model_id=_MODEL_ID)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        ctx = adapter._build_context(df)
        assert ctx.ndim == 1
        np.testing.assert_array_almost_equal(ctx, [1.0, 2.0])

    def test_output_dtype(self):
        adapter = ChronosAdapter(pretrained_model_id=_MODEL_ID)
        df = pd.DataFrame({"v": np.arange(10, dtype=np.float64)})
        ctx = adapter._build_context(df)
        assert ctx.dtype == np.float32


class TestChronosModelLoader:
    def test_loads_via_pipeline(self):
        from tsfb.base.models.model_loader import get_models

        factories = get_models(
            {
                "models": [
                    {
                        "model_name": "foundation.chronos",
                        "model_hyper_params": {
                            "pretrained_model_id": _MODEL_ID,
                            "device": "cpu",
                            "torch_dtype": "float32",
                            "num_samples": 9,
                        },
                    }
                ]
            }
        )
        adapter = factories[0]()
        assert isinstance(adapter, ChronosAdapter)
        assert adapter._num_samples == 9


@pytest.mark.slow
class TestChronosBatchForecast:
    """Tests for ChronosAdapter.batch_forecast() using the BatchMaker protocol."""

    def test_batch_shape_and_dtype(self, chronos_zeroshot):
        from tsfb.base.evaluation.strategy.rolling_batch_maker import (  # noqa: PLC0415
            RollingForecastEvalBatchMaker,
            RollingForecastPredictBatchMaker,
        )

        np.random.seed(0)
        df = pd.DataFrame({"v": np.cumsum(np.random.randn(100)) + 50.0})
        predict_maker = RollingForecastPredictBatchMaker(
            RollingForecastEvalBatchMaker(df, [50, 60, 70])
        )
        result = chronos_zeroshot.batch_forecast(12, predict_maker)
        assert result.shape == (3, 12)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

    def test_batch_exhausts_maker(self, chronos_zeroshot):
        from tsfb.base.evaluation.strategy.rolling_batch_maker import (  # noqa: PLC0415
            RollingForecastEvalBatchMaker,
            RollingForecastPredictBatchMaker,
        )

        np.random.seed(1)
        df = pd.DataFrame({"v": np.cumsum(np.random.randn(100)) + 50.0})
        eval_maker = RollingForecastEvalBatchMaker(df, [50, 60, 70])
        predict_maker = RollingForecastPredictBatchMaker(eval_maker)
        chronos_zeroshot.batch_forecast(6, predict_maker)
        assert not predict_maker.has_more_batches()

    def test_supports_batch_forecast_flag(self):
        assert ChronosAdapter(_MODEL_ID).supports_batch_forecast is True


@pytest.mark.slow
class TestChronosForecastFitFinetuned:
    """Marked slow — fine-tunes a model which takes seconds on CPU."""

    def test_forecast_fit_finetuned_runs(self, long_series, isolate_test_artifacts):
        adapter = ChronosAdapter(
            pretrained_model_id=_MODEL_ID,
            device="cpu",
            inference_mode="finetuned",
            torch_dtype="float32",
            finetune_config={
                "mode": "lora",
                "epochs": 1,
                "learning_rate": 1e-5,
                "batch_size": 8,
                "checkpoint_dir": str(isolate_test_artifacts / "chronos-finetuned"),
            },
        )
        result = adapter.forecast_fit(long_series)
        assert result is adapter
        assert adapter._pipeline is not None

    def test_forecast_after_finetune(self, long_series, isolate_test_artifacts):
        adapter = ChronosAdapter(
            pretrained_model_id=_MODEL_ID,
            device="cpu",
            inference_mode="finetuned",
            torch_dtype="float32",
            finetune_config={
                "mode": "lora",
                "epochs": 1,
                "learning_rate": 1e-5,
                "batch_size": 8,
                "checkpoint_dir": str(isolate_test_artifacts / "chronos-finetuned"),
            },
        )
        adapter.forecast_fit(long_series)
        preds = adapter.forecast(12, long_series)
        assert preds.shape == (12,)
        assert not np.any(np.isnan(preds))
