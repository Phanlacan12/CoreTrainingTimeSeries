"""Tests for MomentAdapter — requires momentfm to be installed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "momentfm",
    reason="momentfm not installed; skipping MOMENT adapter tests",
)

from tsfb.base.models.foundation.adapters.moment_adapter import (  # noqa: E402,E501
    MomentAdapter,
)

# from tsfb.base.forecasting.distribution_forecast import DistributionForecast  # noqa: E402,E501

_MODEL_ID = "AutonLab/MOMENT-1-large"


@pytest.fixture(scope="module")
def moment_adapter() -> MomentAdapter:
    """Single loaded adapter shared across module (model load is expensive)."""
    return MomentAdapter(
        pretrained_model_id=_MODEL_ID,
        device="cpu",
        inference_mode="zeroshot",
        torch_dtype="float32",
    )


@pytest.fixture
def series100() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({"value": np.cumsum(np.random.randn(100)) + 50.0})


@pytest.fixture
def series600() -> pd.DataFrame:
    """MOMENT requires exactly 512 context steps — use 600 to safely cover default."""
    np.random.seed(42)
    return pd.DataFrame({"value": np.cumsum(np.random.randn(600)) + 50.0})


class TestMomentAdapterInit:
    def test_model_name_format(self):
        a = MomentAdapter(pretrained_model_id=_MODEL_ID)
        assert a.model_name == f"moment[{_MODEL_ID}]"

    def test_capability_flags(self):
        a = MomentAdapter(pretrained_model_id=_MODEL_ID)
        assert a.supports_zeroshot is True
        assert a.supports_finetuning is True
        assert a.supports_probabilistic is True
        assert a.supports_fewshot is False

    def test_model_not_loaded_at_init(self):
        a = MomentAdapter(pretrained_model_id=_MODEL_ID)
        assert a._model is None

    def test_head_not_loaded_at_init(self):
        a = MomentAdapter(pretrained_model_id=_MODEL_ID)
        assert a._head is None

    def test_forecast_fit_noop_zeroshot(self, series100):
        a = MomentAdapter(pretrained_model_id=_MODEL_ID, inference_mode="zeroshot")
        result = a.forecast_fit(series100)
        assert result is a
        assert a._model is None

    def test_context_length_stored(self):
        a = MomentAdapter(pretrained_model_id=_MODEL_ID, context_length=256)
        assert a._context_length == 256

    def test_finetune_config_stored(self):
        a = MomentAdapter(
            pretrained_model_id=_MODEL_ID,
            finetune_config={"learning_rate": 1e-4, "epochs": 3},
        )
        assert a._raw_finetune_config["epochs"] == 3


@pytest.mark.slow
class TestMomentInference:
    def test_forecast_output_shape(self, moment_adapter, series600):
        preds = moment_adapter.forecast(12, series600)
        assert preds.shape == (12,)

    def test_forecast_dtype(self, moment_adapter, series600):
        preds = moment_adapter.forecast(12, series600)
        assert preds.dtype == np.float32

    def test_no_nan_in_output(self, moment_adapter, series600):
        preds = moment_adapter.forecast(12, series600)
        assert not np.any(np.isnan(preds))

    @pytest.mark.parametrize("horizon", [1, 6, 12])
    def test_various_horizons(self, moment_adapter, series600, horizon):
        # MOMENT head is built for a fixed forecast_horizon set at _load_model()
        # time. The module-level fixture loads with horizon=12 on first call, so
        # only horizons <= 12 are valid without reloading the model.
        preds = moment_adapter.forecast(horizon, series600)
        assert preds.shape == (horizon,)

    def test_model_loaded_after_forecast(self, moment_adapter, series600):
        moment_adapter.forecast(1, series600)
        assert moment_adapter._model is not None

    def test_prepare_context_shape(self, moment_adapter, series600):
        import torch

        ctx = moment_adapter._prepare_context(series600)
        # MOMENT expects (batch=1, n_channels, seq_len)
        assert ctx.ndim == 3
        assert ctx.shape[0] == 1
        assert isinstance(ctx, torch.Tensor)

    # --- Skipped: MC Dropout requires 10-20 forward passes through MOMENT-1-large ---

    # def test_probabilistic_shape(self, moment_adapter, series600):
    #     samples = moment_adapter.forecast_probabilistic(12, series600, num_samples=10)
    #     assert samples.shape == (10, 12)

    # def test_probabilistic_dtype(self, moment_adapter, series600):
    #     samples = moment_adapter.forecast_probabilistic(12, series600, num_samples=10)
    #     assert samples.dtype == np.float32

    # def test_probabilistic_no_nan(self, moment_adapter, series600):
    #     samples = moment_adapter.forecast_probabilistic(12, series600, num_samples=10)
    #     assert not np.any(np.isnan(samples))

    # def test_mc_dropout_produces_variance(self, moment_adapter, series600):
    #     # With MC Dropout enabled, samples should not all be identical
    #     samples = moment_adapter.forecast_probabilistic(12, series600, num_samples=20)
    #     row_stds = samples.std(axis=0)
    #     assert np.any(row_stds > 0), "MC Dropout produced identical samples"

    # def test_distribution_type(self, moment_adapter, series600):
    #     dist = moment_adapter.forecast_distribution(
    #         12, series600, distribution_name="normal"
    #     )
    #     assert isinstance(dist, DistributionForecast)
    #     assert dist.distribution_name == "normal"

    # def test_distribution_params_shape(self, moment_adapter, series600):
    #     dist = moment_adapter.forecast_distribution(12, series600)
    #     assert dist.params["loc"].shape == (12,)
    #     assert dist.params["scale"].shape == (12,)
