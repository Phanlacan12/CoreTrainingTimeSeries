"""Tests for TiRexAdapter — requires the optional TiRex-2 stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip(
    "tirex2",
    reason="tirex-2 not installed; skipping TiRex adapter tests",
)

from tsfb.base.forecasting.distribution_forecast import (  # noqa: E402,E501
    DistributionForecast,
)
from tsfb.base.models.foundation.adapters.tirex_adapter import (  # noqa: E402,E501
    TiRexAdapter,
)
from tsfb.base.models.foundation.registry import get_foundation_spec  # noqa: E402

_MODEL_ID = "NX-AI/TiRex-2"


@pytest.fixture(scope="module")
def tirex_target_only() -> TiRexAdapter:
    return TiRexAdapter(
        pretrained_model_id=_MODEL_ID,
        device="cpu",
        use_covariates=False,
    )


@pytest.fixture(scope="module")
def tirex_with_covariates() -> TiRexAdapter:
    return TiRexAdapter(
        pretrained_model_id=_MODEL_ID,
        device="cpu",
        use_covariates=True,
    )


@pytest.fixture
def series96() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=96, freq="h")
    values = np.sin(np.arange(96, dtype=np.float32) / 8.0)
    return pd.DataFrame({"target": values}, index=index)


class TestTiRexAdapterInit:
    def test_capabilities_and_registry(self):
        adapter = TiRexAdapter(device="cpu")
        spec = get_foundation_spec("tirex")
        assert adapter.model_name == f"tirex[{_MODEL_ID}]"
        assert adapter.supports_zeroshot is True
        assert adapter.supports_probabilistic is True
        assert adapter.supports_distribution is True
        assert adapter.supports_finetuning is False
        assert adapter.supports_fewshot is False
        assert spec.default_model_id == _MODEL_ID
        assert spec.supports_probabilistic is True
        assert spec.supports_finetuning is False

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"inference_mode": "finetuned"}, "only inference_mode='zeroshot'"),
            ({"context_length": 96}, "does not expose context_length"),
            ({"cross_learning": True}, "has no cross_learning option"),
        ],
    )
    def test_unsupported_modes_fail_early(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TiRexAdapter(device="cpu", **kwargs)


@pytest.mark.slow
class TestTiRexForecast:
    def test_point_forecast_shape_dtype_and_values(
        self, tirex_target_only, series96
    ):
        result = tirex_target_only.forecast(24, series96)
        assert result.shape == (24,)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert tirex_target_only._pipeline is not None

    def test_short_and_leading_missing_context(self, tirex_target_only, series96):
        short = series96.tail(12)
        padded_index = pd.date_range(
            series96.index[0] - pd.Timedelta(hours=4),
            periods=len(series96) + 4,
            freq="h",
        )
        padded = pd.DataFrame(
            {
                "target": np.concatenate(
                    [
                        np.full(4, np.nan, dtype=np.float32),
                        series96["target"].to_numpy(),
                    ]
                )
            },
            index=padded_index,
        )
        assert tirex_target_only.forecast(3, short).shape == (3,)
        assert tirex_target_only.forecast(3, padded).shape == (3,)

    def test_future_covariates(self, tirex_with_covariates, series96):
        horizon = 12
        full_index = pd.date_range(
            series96.index[0], periods=len(series96) + horizon, freq="h"
        )
        covariates = {
            "future_covariates": pd.DataFrame(
                {"known": np.cos(np.arange(len(full_index)) / 12.0)},
                index=full_index,
            )
        }
        result = tirex_with_covariates.forecast(
            horizon, series96, covariates=covariates
        )
        assert result.shape == (horizon,)
        assert np.isfinite(result).all()

    def test_probabilistic_and_distribution_outputs(
        self, tirex_target_only, series96
    ):
        samples = tirex_target_only.forecast_probabilistic(
            4, series96, num_samples=13
        )
        distribution = tirex_target_only.forecast_distribution(4, series96)
        assert samples.shape == (13, 4)
        assert np.isfinite(samples).all()
        assert isinstance(distribution, DistributionForecast)
        assert distribution.prediction_length == 4

    @pytest.mark.parametrize(
        "series",
        [
            pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}),
            pd.DataFrame({"target": [1.0, np.nan]}),
            pd.DataFrame({"target": ["a", "b"]}),
        ],
    )
    def test_invalid_targets_fail(self, tirex_target_only, series):
        with pytest.raises(ValueError):
            tirex_target_only.forecast(2, series)

    def test_short_future_covariate_fails(
        self, tirex_with_covariates, series96
    ):
        short_future = pd.DataFrame(
            {"known": np.arange(len(series96) + 1, dtype=float)},
            index=pd.date_range(
                series96.index[0], periods=len(series96) + 1, freq="h"
            ),
        )
        with pytest.raises(ValueError, match="does not cover"):
            tirex_with_covariates.forecast(
                2,
                series96,
                covariates={"future_covariates": short_future},
            )

    def test_native_horizon_limit(self, tirex_target_only, series96):
        with pytest.raises(ValueError, match="horizon <= 320"):
            tirex_target_only.forecast(321, series96)


def test_both_coldstart_configs_expose_the_same_tirex_arms():
    root = Path(__file__).resolve().parents[4]
    expected = {"tirex_cov_off", "tirex_cov_on"}
    for filename in ("coldstart_default.yaml", "coldstart_static_grouped.yaml"):
        with open(root / "tsfb_configs" / filename, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        assert expected.issubset(config["model"])
        for name in expected:
            arm = config["model"][name]
            assert arm["approach"] == "univariate_per_series"
            assert arm["model_name"] == "foundation.tirex"
        assert config["model"]["tirex_cov_off"]["model_hyper_params"][
            "use_covariates"
        ] is False
        assert config["model"]["tirex_cov_on"]["model_hyper_params"][
            "use_covariates"
        ] is True
        assert "seasonal_naive" in config["model"]
