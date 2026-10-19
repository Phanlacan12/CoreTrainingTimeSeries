"""Tests for MoiraiAdapter — requires uni2ts to be installed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# uni2ts may be installed but its model.moirai submodule missing — check the
# exact import the adapter uses so we skip correctly in both cases.
pytest.importorskip(
    "uni2ts.model.moirai",
    reason="uni2ts.model.moirai not available; skipping Moirai adapter tests",
)

from tsfb.base.forecasting.distribution_forecast import (  # noqa: E402,E501
    DistributionForecast,
)
from tsfb.base.models.foundation.adapters.moirai_adapter import (  # noqa: E402,E501
    MoiraiAdapter,
)

_MODEL_ID = "Salesforce/moirai-1.0-R-small"


@pytest.fixture(scope="module")
def moirai_adapter() -> MoiraiAdapter:
    """Single loaded adapter shared across module (model load is expensive)."""
    return MoiraiAdapter(
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


class TestMoiraiAdapterInit:
    def test_model_name_format(self):
        a = MoiraiAdapter(pretrained_model_id=_MODEL_ID)
        assert a.model_name == f"moirai[{_MODEL_ID}]"

    def test_capability_flags(self):
        a = MoiraiAdapter(pretrained_model_id=_MODEL_ID)
        assert a.supports_zeroshot is True
        assert a.supports_finetuning is True
        assert a.supports_probabilistic is True
        assert a.supports_fewshot is False

    def test_pipeline_not_loaded_at_init(self):
        a = MoiraiAdapter(pretrained_model_id=_MODEL_ID)
        assert a._pipeline is None

    def test_num_samples_stored(self):
        a = MoiraiAdapter(pretrained_model_id=_MODEL_ID, num_samples=20)
        assert a._num_samples == 20

    def test_forecast_fit_noop_zeroshot(self, series100):
        a = MoiraiAdapter(pretrained_model_id=_MODEL_ID, inference_mode="zeroshot")
        result = a.forecast_fit(series100)
        assert result is a
        assert a._pipeline is None

    def test_finetune_config_stored(self):
        a = MoiraiAdapter(
            pretrained_model_id=_MODEL_ID,
            finetune_config={"mode": "lora", "epochs": 1},
        )
        assert a._raw_finetune_config["mode"] == "lora"


@pytest.mark.slow
class TestMoiraiInference:
    def test_forecast_output_shape(self, moirai_adapter, series100):
        preds = moirai_adapter.forecast(12, series100)
        assert preds.shape == (12,)

    def test_forecast_dtype(self, moirai_adapter, series100):
        preds = moirai_adapter.forecast(12, series100)
        assert preds.dtype == np.float32

    def test_no_nan_in_output(self, moirai_adapter, series100):
        preds = moirai_adapter.forecast(12, series100)
        assert not np.any(np.isnan(preds))

    @pytest.mark.parametrize("horizon", [1, 6, 24])
    def test_various_horizons(self, moirai_adapter, series100, horizon):
        preds = moirai_adapter.forecast(horizon, series100)
        assert preds.shape == (horizon,)

    def test_pipeline_loaded_after_forecast(self, moirai_adapter, series100):
        moirai_adapter.forecast(1, series100)
        assert moirai_adapter._pipeline is not None

    def test_probabilistic_shape(self, moirai_adapter, series100):
        samples = moirai_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert samples.shape == (9, 12)

    def test_probabilistic_dtype(self, moirai_adapter, series100):
        samples = moirai_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert samples.dtype == np.float32

    def test_probabilistic_no_nan(self, moirai_adapter, series100):
        samples = moirai_adapter.forecast_probabilistic(12, series100, num_samples=9)
        assert not np.any(np.isnan(samples))

    @pytest.mark.parametrize("num_samples", [1, 9, 18])
    def test_num_samples_parameter(self, moirai_adapter, series100, num_samples):
        samples = moirai_adapter.forecast_probabilistic(
            12, series100, num_samples=num_samples
        )
        assert samples.shape[0] == num_samples

    def test_distribution_type(self, moirai_adapter, series100):
        dist = moirai_adapter.forecast_distribution(
            12, series100, distribution_name="normal"
        )
        assert isinstance(dist, DistributionForecast)
        assert dist.distribution_name == "normal"

    def test_distribution_params_shape(self, moirai_adapter, series100):
        dist = moirai_adapter.forecast_distribution(12, series100)
        assert dist.params["loc"].shape == (12,)
        assert dist.params["scale"].shape == (12,)

    def test_point_forecast_is_median_of_samples(self, moirai_adapter, series100):
        samples = moirai_adapter.forecast_probabilistic(12, series100, num_samples=9)
        preds = moirai_adapter.forecast(12, series100)
        # Point forecast = median of samples — should be in same order of magnitude
        assert np.allclose(preds, np.median(samples, axis=0), rtol=0.1)
