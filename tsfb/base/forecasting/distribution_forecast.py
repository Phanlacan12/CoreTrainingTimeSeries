"""
DistributionForecast — parametric distribution wrapper for time series forecasts.

Mirrors GluonTS DistributionForecast: wraps per-timestep scipy distribution
parameters and exposes .quantile(), .mean, .variance, .sample(), .log_prob().

Supports a fallback path: fit a Normal distribution to an existing sample
matrix via moment-matching (from_samples classmethod). This lets adapters
that currently expose samples() upgrade to distribution output without
changing their forecast_probabilistic() implementation.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from tsfb.base.forecasting.base_forecast import ForecastBase

# ---------------------------------------------------------------------------
# Scipy distribution dispatcher
# ---------------------------------------------------------------------------


def _scipy_dist(name: str):
    """Return the scipy.stats frozen-distribution constructor for *name*."""
    mapping = {
        "normal": stats.norm,
        "student_t": stats.t,
        "poisson": stats.poisson,
        "negative_binomial": stats.nbinom,
    }
    if name not in mapping:
        raise ValueError(f"Unknown distribution '{name}'. Available: {list(mapping)}")
    return mapping[name]


def _make_frozen(name: str, params: dict):
    """
    Build a per-timestep scipy frozen distribution array.

    Returns a callable that accepts an array of quantile levels and
    returns shape (n_quantiles, T).
    """
    dist = _scipy_dist(name)
    if name == "normal":
        return dist(loc=params["loc"], scale=np.clip(params["scale"], 1e-6, None))
    if name == "student_t":
        return dist(
            df=np.clip(params["df"], 2.01, None),
            loc=params["loc"],
            scale=np.clip(params["scale"], 1e-6, None),
        )
    if name == "poisson":
        return dist(mu=np.clip(params["mu"], 1e-6, None))
    if name == "negative_binomial":
        mu = np.clip(params["mu"], 1e-6, None)
        alpha = np.clip(params["alpha"], 1e-6, None)
        # scipy NegBinomial parameterisation: n=1/alpha, p=1/(1+alpha*mu)
        n = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        return dist(n=n, p=p)
    raise ValueError(f"Unhandled distribution in _make_frozen: '{name}'")


# ---------------------------------------------------------------------------
# DistributionForecast
# ---------------------------------------------------------------------------


class DistributionForecast(ForecastBase):
    """
    GluonTS DistributionForecast-compatible wrapper.

    Holds a parametric distribution per timestep (scipy-backed) and
    exposes the same interface as ProbabilisticForecast so strategy /
    metric code is fully agnostic.

    Usage::

        df = DistributionForecast(
            "normal",
            {"loc": np.zeros(24), "scale": np.ones(24)},
        )
        q10 = df.quantile(0.1)          # (24,)
        samples = df.sample(100)        # (100, 24)
        ll = df.log_prob(y_true)        # (24,)
    """

    def __init__(self, distribution_name: str, params: dict) -> None:
        """
        Args:
            distribution_name: 'normal' | 'student_t' | 'negative_binomial'
            params: {param_name: np.ndarray of shape (prediction_length,)}
        """
        self.distribution_name = distribution_name
        self.params = {k: np.asarray(v, dtype=float) for k, v in params.items()}
        self._frozen = _make_frozen(distribution_name, self.params)

        lengths = {v.shape[0] for v in self.params.values()}
        if len(lengths) != 1:
            raise ValueError("All params must have the same prediction_length.")
        self._prediction_length = lengths.pop()

    # ------------------------------------------------------------------
    # ForecastBase interface
    # ------------------------------------------------------------------

    @property
    def prediction_length(self) -> int:
        """Number of forecast steps."""
        return self._prediction_length

    def quantile(self, q: Union[float, list]) -> np.ndarray:
        """
        Compute per-timestep quantile(s) via the distribution ppf.

        Args:
            q: Single float or list of floats in [0, 1].

        Returns:
            Shape (prediction_length,) for scalar q;
            (len(q), prediction_length) for list q.
        """
        scalar = isinstance(q, (int, float))
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.stack([self._frozen.ppf(qi) for qi in q_arr])  # (len(q), T)
        return result[0] if scalar else result

    @property
    def mean(self) -> np.ndarray:
        """Distribution mean — shape (prediction_length,)."""
        return np.asarray(self._frozen.mean(), dtype=np.float32)

    @property
    def median(self) -> np.ndarray:
        """Distribution median — shape (prediction_length,)."""
        return np.asarray(self._frozen.median(), dtype=np.float32)

    # ------------------------------------------------------------------
    # Distribution-specific extras (not in ProbabilisticForecast)
    # ------------------------------------------------------------------

    @property
    def variance(self) -> np.ndarray:
        """Distribution variance — shape (prediction_length,)."""
        return np.asarray(self._frozen.var(), dtype=np.float32)

    @property
    def std(self) -> np.ndarray:
        """Distribution standard deviation — shape (prediction_length,)."""
        return np.sqrt(self.variance)

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Draw IID samples from the distribution.

        Returns:
            Shape (num_samples, prediction_length).
        """
        # Each _frozen.rvs() call returns shape (prediction_length,) —
        # one draw per timestep. Stack into (num_samples, prediction_length).
        return np.stack([self._frozen.rvs() for _ in range(num_samples)]).astype(
            np.float32
        )

    def log_prob(self, y_true: np.ndarray) -> np.ndarray:
        """
        Pointwise log-likelihood of observed values under the distribution.

        Args:
            y_true: Observed values, shape (prediction_length,).

        Returns:
            Log-likelihood per timestep, shape (prediction_length,).
        """
        return np.asarray(self._frozen.logpdf(y_true), dtype=np.float32)

    def to_payload_dict(self) -> dict:
        """Serialize distribution params for interval_payload JSON."""
        return {
            "type": self.distribution_name,
            **{k: v.tolist() for k, v in self.params.items()},
        }

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        distribution_name: str = "normal",
    ) -> "DistributionForecast":
        """
        Fit a parametric distribution to an existing sample matrix.

        Uses moment-matching (mean + std for Normal/StudentT) so that
        adapters returning raw samples can be 'upgraded' to distribution
        output without changing their forecast_probabilistic() code.

        Args:
            samples: Shape (num_samples, prediction_length).
            distribution_name: Distribution family to fit.

        Returns:
            DistributionForecast fitted to the samples.
        """
        if samples.ndim != 2:
            raise ValueError(
                "from_samples expects shape (num_samples, prediction_length)"
            )
        loc = np.mean(samples, axis=0)
        scale = np.std(samples, axis=0) + 1e-6

        if distribution_name == "normal":
            return cls("normal", {"loc": loc, "scale": scale})

        if distribution_name == "student_t":
            # Estimate df via kurtosis: excess_kurtosis = 6/(df-4) → df = 6/ek + 4
            kurt = np.clip(
                np.mean(((samples - loc) / scale) ** 4, axis=0) - 3, 0.01, None
            )
            df = np.clip(6.0 / kurt + 4, 2.1, 30.0)
            return cls("student_t", {"df": df, "loc": loc, "scale": scale})

        if distribution_name == "poisson":
            mu = np.clip(np.mean(np.abs(samples), axis=0), 1e-6, None)
            return cls("poisson", {"mu": mu.astype(np.float32)})

        if distribution_name == "negative_binomial":
            mu = np.clip(loc, 1e-6, None)
            var = np.clip(np.var(samples, axis=0), mu + 1e-6, None)
            alpha = (var - mu) / (mu**2)
            return cls(
                "negative_binomial", {"mu": mu, "alpha": np.clip(alpha, 1e-6, None)}
            )

        raise ValueError(
            f"Unsupported distribution for from_samples: {distribution_name}"
        )

    # ------------------------------------------------------------------
    # to_quantile_dfs — override to add distribution column
    # ------------------------------------------------------------------

    def to_quantile_dfs(
        self,
        levels: list,
        index: pd.Index,
        target_cols: list,
    ) -> dict:
        """
        Export quantile forecasts as DataFrames (same interface as ForecastBase).

        Args:
            levels: Quantile levels, e.g. [0.1, 0.5, 0.9].
            index: DatetimeIndex or RangeIndex for forecast horizon.
            target_cols: Column names (univariate → single-element list).

        Returns:
            {"q10": pd.DataFrame, ...}
        """
        result = {}
        for q in levels:
            key = f"q{int(round(q * 100)):02d}"
            values = self.quantile(q)[:, np.newaxis]
            result[key] = pd.DataFrame(values, index=index, columns=target_cols)
        return result
