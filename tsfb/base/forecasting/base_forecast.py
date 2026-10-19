from __future__ import annotations

import abc
from typing import Union

import numpy as np
import pandas as pd


class ForecastBase(abc.ABC):
    """
    Abstract base for all probabilistic forecast wrappers.

    Mirrors the GluonTS Forecast interface so downstream strategy and
    metric code is agnostic to whether the output came from samples,
    quantiles, or a parametric distribution.
    """

    @abc.abstractmethod
    def quantile(self, q: Union[float, list]) -> np.ndarray:
        """
        Compute quantile(s).

        Args:
            q: Single float or list of floats in [0, 1].

        Returns:
            Shape (prediction_length,) for scalar q;
            (len(q), prediction_length) for list q.
        """

    @property
    @abc.abstractmethod
    def mean(self) -> np.ndarray:
        """Mean forecast — shape (prediction_length,)."""

    @property
    def median(self) -> np.ndarray:
        """Median forecast — defaults to quantile(0.5)."""
        return self.quantile(0.5)

    @property
    @abc.abstractmethod
    def prediction_length(self) -> int:
        """Number of forecast steps."""

    def to_quantile_dfs(
        self,
        levels: list,
        index: pd.Index,
        target_cols: list,
    ) -> dict:
        """
        Export quantile forecasts as a dict of DataFrames.

        Args:
            levels: Quantile levels, e.g. [0.1, 0.5, 0.9].
            index: DatetimeIndex or RangeIndex for forecast horizon.
            target_cols: Column names for the target variables.

        Returns:
            {"q10": pd.DataFrame, "q50": pd.DataFrame, ...}
        """
        result = {}
        for q in levels:
            key = f"q{int(round(q * 100)):02d}"
            values = self.quantile(q)
            if values.ndim == 1:
                values = values[:, np.newaxis]
            result[key] = pd.DataFrame(values, index=index, columns=target_cols)
        return result
