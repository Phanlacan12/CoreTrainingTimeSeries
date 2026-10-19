"""
Abstract base class for multi-modal foundation models.

Multi-modal models combine time series with one or more additional modalities
(text, image, tabular) to produce richer forecasts. This module provides the
interface contract so that future implementations do not require a refactor of
the adapter or pipeline layers.

Status: PLACEHOLDER — no concrete implementations yet.
"""
from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np
import pandas as pd

from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.models.foundation.multimodal.context_types import (
    ImageContext,
    TabularContext,
    TextContext,
)


class MultiModalFoundationBase(FoundationModelBase, metaclass=abc.ABCMeta):
    """
    Abstract base for foundation models that consume multiple input modalities.

    Subclasses must implement forecast_with_context(), which extends the
    standard forecast() signature with optional text / image / tabular inputs.

    The standard forecast() delegates to forecast_with_context() with all
    context args set to None, so the ModelBase pipeline remains compatible.
    """

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        return self.forecast_with_context(
            horizon=horizon,
            series=series,
            covariates=covariates,
        )

    @abc.abstractmethod
    def forecast_with_context(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        text_context: Optional[List[TextContext]] = None,
        image_context: Optional[List[ImageContext]] = None,
        tabular_context: Optional[TabularContext] = None,
    ) -> np.ndarray:
        """
        Produce a forecast conditioned on time series plus optional modalities.

        Implementations should gracefully handle the case where some or all
        context arguments are None (i.e., fall back to univariate forecasting).

        Parameters
        ----------
        horizon:
            Number of steps to forecast.
        series:
            Historical time series values.
        covariates:
            Standard TSFB covariate dict (past/future/static).
        text_context:
            List of TextContext objects aligned with the context window.
        image_context:
            List of ImageContext frames aligned with the context window.
        tabular_context:
            External feature DataFrame aligned with the context window.

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
