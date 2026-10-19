"""Tests for FoundationModelBase ABC and capability flags."""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.base_model import ModelBase
from tsfb.base.models.foundation.base_foundation import FoundationModelBase


class _MinimalFoundation(FoundationModelBase):
    """Minimal concrete subclass — only implements required abstract methods."""

    @property
    def model_name(self) -> str:
        return "minimal_foundation"

    def forecast(self, horizon, series, *, covariates=None) -> np.ndarray:
        return np.zeros(horizon, dtype=np.float32)


class _ProbabilisticFoundation(FoundationModelBase):
    @property
    def model_name(self) -> str:
        return "probabilistic_foundation"

    def forecast(self, horizon, series, *, covariates=None) -> np.ndarray:
        return np.ones(horizon, dtype=np.float32)

    @property
    def supports_probabilistic(self) -> bool:
        return True

    def forecast_probabilistic(
        self, horizon, series, num_samples=100, *, covariates=None
    ):
        return np.random.randn(num_samples, horizon).astype(np.float32)


class TestFoundationModelBaseInheritance:
    def test_is_model_base_subclass(self):
        assert issubclass(FoundationModelBase, ModelBase)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            FoundationModelBase()

    def test_concrete_subclass_instantiates(self):
        m = _MinimalFoundation()
        assert isinstance(m, FoundationModelBase)
        assert isinstance(m, ModelBase)


class TestDefaultCapabilityFlags:
    def setup_method(self):
        self.model = _MinimalFoundation()

    def test_supports_zeroshot_default_true(self):
        assert self.model.supports_zeroshot is True

    def test_supports_finetuning_default_false(self):
        assert self.model.supports_finetuning is False

    def test_supports_probabilistic_default_false(self):
        assert self.model.supports_probabilistic is False

    def test_supports_fewshot_default_false(self):
        assert self.model.supports_fewshot is False


class TestForecastFitNoOp:
    def test_forecast_fit_returns_self(self, short_series):
        model = _MinimalFoundation()
        result = model.forecast_fit(short_series)
        assert result is model

    def test_forecast_fit_with_covariates(self, short_series):
        model = _MinimalFoundation()
        result = model.forecast_fit(short_series, covariates={"past": short_series})
        assert result is model


class TestForecastOutput:
    def test_forecast_shape(self, short_series):
        model = _MinimalFoundation()
        horizon = 12
        preds = model.forecast(horizon, short_series)
        assert preds.shape == (horizon,)

    def test_forecast_dtype(self, short_series):
        model = _MinimalFoundation()
        preds = model.forecast(12, short_series)
        assert preds.dtype == np.float32


class TestOptionalMethodsRaiseWhenNotImplemented:
    def setup_method(self):
        self.model = _MinimalFoundation()
        self.series = pd.DataFrame({"v": np.arange(50, dtype=float)})

    def test_finetune_raises(self):
        from tsfb.base.schema.foundation_config import FineTuneConfig

        with pytest.raises(NotImplementedError, match="fine-tuning"):
            self.model.finetune(self.series, FineTuneConfig())

    def test_forecast_probabilistic_raises(self):
        with pytest.raises(NotImplementedError, match="probabilistic"):
            self.model.forecast_probabilistic(12, self.series)

    def test_set_few_shot_context_raises(self):
        with pytest.raises(NotImplementedError, match="few-shot"):
            self.model.set_few_shot_context(self.series)


class TestProbabilisticFoundation:
    def test_probabilistic_output_shape(self, short_series):
        model = _ProbabilisticFoundation()
        samples = model.forecast_probabilistic(12, short_series, num_samples=50)
        assert samples.shape == (50, 12)

    def test_supports_probabilistic_flag(self):
        model = _ProbabilisticFoundation()
        assert model.supports_probabilistic is True


class TestSeriesToTensor:
    def test_single_column(self):
        model = _MinimalFoundation()
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        arr = model._series_to_tensor(df)
        assert arr.shape == (3, 1)
        assert arr.dtype == np.float32

    def test_multi_column(self):
        model = _MinimalFoundation()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        arr = model._series_to_tensor(df)
        assert arr.shape == (2, 2)

    def test_ignores_non_numeric(self):
        model = _MinimalFoundation()
        df = pd.DataFrame({"v": [1.0, 2.0], "label": ["x", "y"]})
        arr = model._series_to_tensor(df)
        assert arr.shape == (2, 1)

    def test_repr(self):
        model = _MinimalFoundation()
        assert repr(model) == "minimal_foundation"
