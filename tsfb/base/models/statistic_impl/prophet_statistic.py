from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from scipy.stats import norm as _scipy_norm

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.base_model import BatchMaker
from tsfb.base.models.base_statistic import BaseStatistic


class ProphetStatistic(BaseStatistic):
    """Thin wrapper around `prophet.Prophet` for univariate forecasting."""

    def __init__(self, alpha: float = 0.05, **prophet_kwargs):
        """Init with significance level `alpha` (interval_width = 1 - alpha)."""
        super().__init__(alpha=alpha)
        interval_width = max(0.0, min(1.0, 1.0 - float(alpha)))
        self._prophet = Prophet(interval_width=interval_width, **prophet_kwargs)

    @staticmethod
    def required_hyper_params() -> dict:
        """Declare supported hyper-parameters (exposes `alpha`)."""
        return {"alpha": 0.05}

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ProphetStatistic":
        """Fit Prophet on a univariate DataFrame with DatetimeIndex."""
        target_col = self._check_univariate(train_data)
        self._target_col = target_col
        self._t0 = train_data.index[0]

        df = train_data.copy()
        df["t"] = (df.index - self._t0).days.astype(float)
        self._last_train = df

        df_fit = pd.DataFrame(
            {"ds": train_data.index.to_pydatetime(), "y": train_data[target_col].values}
        )

        self._prophet.fit(df_fit)
        self._results = self._prophet
        return self

    def _infer_freq(
        self, index: pd.DatetimeIndex
    ) -> pd.tseries.offsets.BaseOffset | str:
        """
        Infer frequency from index: try pd.infer_freq; if irregular,
        use the most frequent positive delta (mode); otherwise return 'D'.
        """
        freq = pd.infer_freq(index)
        if freq is not None:
            return freq

        if len(index) >= 2:
            deltas_ns = np.diff(index.view("int64"))
            deltas_ns = deltas_ns[deltas_ns > 0]
            if deltas_ns.size:
                unique_vals, counts = np.unique(deltas_ns, return_counts=True)
                max_count = counts.max()
                candidates = unique_vals[counts == max_count]
                chosen_ns = int(candidates.min())
                return pd.to_timedelta(chosen_ns, unit="ns")

        return "D"

    def _future_from_origin(
        self, origin_index: pd.DatetimeIndex, horizon: int
    ) -> pd.DataFrame:
        """Build future `ds` dates starting after `origin_index[-1]`."""
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        freq = self._infer_freq(origin_index)
        start = origin_index[-1] + (
            pd.tseries.frequencies.to_offset(freq) if isinstance(freq, str) else freq
        )
        future_ds = pd.date_range(start=start, periods=horizon, freq=freq)
        return pd.DataFrame({"ds": future_ds})

    def forecast_with_ci(self, horizon: int) -> dict:
        """Predict `horizon` steps ahead and return mean and PI bounds."""
        if self._results is None or self._last_train is None:
            raise RuntimeError("Model has not been fitted.")

        origin_index = self._last_train.index
        future_df = self._future_from_origin(origin_index, horizon)
        fc = self._prophet.predict(future_df)

        mu_hat = fc["yhat"].to_numpy(dtype=float)
        pred_lo = fc["yhat_lower"].to_numpy(dtype=float)
        pred_hi = fc["yhat_upper"].to_numpy(dtype=float)
        y_mode = mu_hat.copy()

        n = len(mu_hat)
        ci_na = np.full(n, np.nan, dtype=float)

        return {
            "mu_hat": mu_hat,
            "ci_lower": ci_na,
            "ci_upper": ci_na,
            "pred_lower": pred_lo,
            "pred_upper": pred_hi,
            "y_mode": y_mode,
        }

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """Predict `horizon` steps ahead for a new series window."""
        if self._results is None:
            raise RuntimeError("Model has not been fitted.")
        if not isinstance(series.index, pd.DatetimeIndex) or len(series) == 0:
            raise ValueError("`series` must be non-empty with a DatetimeIndex.")

        future_df = self._future_from_origin(series.index, horizon)
        fc = self._prophet.predict(future_df)
        return fc["yhat"].to_numpy(dtype=float)

    def _prophet_intervals(self, series: pd.DataFrame, horizon: int) -> tuple:
        """Run Prophet predict() and return (mu, sigma) derived from its intervals."""
        if self._results is None:
            raise RuntimeError("Model has not been fitted.")
        future_df = self._future_from_origin(series.index, horizon)
        fc = self._prophet.predict(future_df)
        mu = fc["yhat"].to_numpy(dtype=np.float32)
        pred_lo = fc["yhat_lower"].to_numpy(dtype=float)
        pred_hi = fc["yhat_upper"].to_numpy(dtype=float)
        # Prophet interval_width = 1 - alpha → z = ppf(1 - alpha/2)
        z = float(_scipy_norm.ppf(1.0 - self._alpha / 2))
        sigma = np.clip((pred_hi - pred_lo) / (2.0 * z), 1e-6, None).astype(np.float32)
        return mu, sigma

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw Normal samples fitted to Prophet's uncertainty intervals."""
        mu, sigma = self._prophet_intervals(series, horizon)
        rng = np.random.default_rng()
        return rng.normal(loc=mu, scale=sigma, size=(num_samples, horizon)).astype(
            np.float32
        )

    def forecast_distribution(  # pylint: disable=too-many-arguments,unused-argument
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> DistributionForecast:
        """Return Normal distribution fitted to Prophet uncertainty intervals."""
        mu, sigma = self._prophet_intervals(series, horizon)
        return DistributionForecast("normal", {"loc": mu, "scale": sigma})

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """Batch forecasting for rolling/backtesting (not implemented)."""
        raise NotImplementedError("Not implemented batch forecasting!")
