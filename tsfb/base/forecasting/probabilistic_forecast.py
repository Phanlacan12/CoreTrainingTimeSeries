from __future__ import annotations

from typing import Union

import numpy as np

from tsfb.base.forecasting.base_forecast import ForecastBase


class ProbabilisticForecast(ForecastBase):
    """
    GluonTS SampleForecast-compatible wrapper for sample-based probabilistic forecasts.

    Wraps a sample matrix and exposes .quantile(), .mean, .median — matching the
    GluonTS SampleForecast interface so downstream code stays model-agnostic.

    Supports univariate: samples shape (num_samples, prediction_length)
    Supports multivariate: samples shape (num_samples, prediction_length, n_targets)
    """

    def __init__(self, samples: np.ndarray) -> None:
        if samples.ndim not in (2, 3):
            raise ValueError(
                f"samples must be 2-D (num_samples, horizon) or "
                f"3-D (num_samples, horizon, n_targets), got shape {samples.shape}"
            )
        self.samples = samples.astype(np.float32)

    @property
    def prediction_length(self) -> int:
        """Number of forecast steps."""
        return self.samples.shape[1]

    @property
    def num_samples(self) -> int:
        """Number of sample trajectories."""
        return self.samples.shape[0]

    @property
    def n_targets(self) -> int:
        """Number of target variables (1 for univariate)."""
        return self.samples.shape[2] if self.samples.ndim == 3 else 1

    def quantile(self, q: Union[float, list]) -> np.ndarray:
        """
        Compute quantile(s) across the sample axis.

        Args:
            q: Single float or list of floats in [0, 1].

        Returns:
            Shape (prediction_length,) for scalar q;
            (len(q), prediction_length) for list q.
            For multivariate: (prediction_length, n_targets)
            or (len(q), prediction_length, n_targets).
        """
        scalar = isinstance(q, (int, float))
        q_arr = np.atleast_1d(np.asarray(q, dtype=float)) * 100
        result: np.ndarray = np.percentile(self.samples, q_arr, axis=0)
        return result[0] if scalar else result

    @property
    def mean(self) -> np.ndarray:
        """Mean — shape (prediction_length,) or (prediction_length, n_targets)."""
        return np.mean(self.samples, axis=0)

    @property
    def median(self) -> np.ndarray:
        """Median — shape (prediction_length,) or (prediction_length, n_targets)."""
        return np.median(self.samples, axis=0)
