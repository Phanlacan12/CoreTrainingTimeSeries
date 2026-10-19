from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.base_model import BatchMaker, ModelBase


class BaseStatistic(ModelBase, metaclass=abc.ABCMeta):
    """
    Base class for statistical univariate time series models.

    This class provides:
      - Input validation for univariate time series with DatetimeIndex.
      - Utilities for generating future time indices based on training data.
      - Forecasting methods that return point forecasts and confidence intervals (CI).
      - An extensible interface for subclasses to implement prediction intervals (PI).

    Notes
    -----
    - Subclasses are responsible for implementing the fitting logic
      (`forecast_fit` or equivalent).
      During fitting, subclasses must set:
        * self._results: fitted model results (e.g., statsmodels results object)
        * self._last_train: training DataFrame with an added 't' column
        * self._t0: the start timestamp of the training data
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize the base class.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level used to compute confidence intervals.
        """
        self._alpha: float = float(alpha)
        self._results = None
        self._last_train: Optional[pd.DataFrame] = None
        self._t0: Optional[pd.Timestamp] = None
        self._target_col: Optional[str] = None

    @property
    def model_name(self) -> str:
        """
        Return the name of the model.

        Returns
        -------
        str
            The class name of the model.
        """
        return self.__class__.__name__

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by the model.

        Returns
        -------
        dict
            Dictionary of required hyperparameters. By default, only "alpha".
        """
        return {"alpha": 0.05}

    def _check_univariate(self, train_data: pd.DataFrame) -> str:
        """
        Validate that the input data is univariate with a DatetimeIndex.

        Parameters
        ----------
        train_data : pd.DataFrame
            Input time series data.

        Returns
        -------
        str
            The name of the target column.
        """
        if not isinstance(train_data.index, pd.DatetimeIndex):
            raise ValueError("train_data must have a DatetimeIndex.")
        if train_data.shape[1] != 1:
            raise ValueError(
                f"{self.model_name} only supports univariate data, "
                f"but got {train_data.shape[1]} columns."
            )
        return train_data.columns[0]

    def _future_time_index(self, horizon: int) -> np.ndarray:
        """
        Generate future time steps as integers (days since _t0).

        Parameters
        ----------
        horizon : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Array of future time indices.
        """
        if self._last_train is None:
            raise ValueError("Please fit the model first (missing _last_train).")
        t_last = float(self._last_train["t"].iloc[-1])
        return np.arange(t_last + 1, t_last + horizon + 1, dtype=float)

    def _compute_mean_and_ci(
        self,
        horizon: int,
        series: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean forecasts and confidence intervals for the mean.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        series : pd.DataFrame, optional
            Reference time series. If provided, the last timestamp of this series
            determines the forecast starting point. Otherwise, the last training
            observation is used.
        covariates : dict, optional
            Additional covariates (not used in the base implementation).

        Returns
        -------
        tuple of np.ndarray
            (mu_hat, ci_lo, ci_hi)
        """
        if self._results is None or self._t0 is None:
            raise RuntimeError("Model has not been fitted.")

        if series is not None:
            if not isinstance(series.index, pd.DatetimeIndex):
                raise ValueError("`series` must have a DatetimeIndex.")
            if len(series) == 0:
                raise ValueError("`series` is empty; need at least one row.")
            t_last = (series.index[-1] - self._t0).days
        else:
            if self._last_train is None:
                raise RuntimeError("Missing _last_train to infer forecast origin.")
            t_last = float(self._last_train["t"].iloc[-1])

        t_future = np.arange(t_last + 1, t_last + horizon + 1, dtype=float)
        x_future = np.column_stack([np.ones_like(t_future), t_future])

        if hasattr(self._results, "get_prediction"):
            pred = self._results.get_prediction(exog=x_future)
            sf = pred.summary_frame(alpha=self._alpha)

            if "mean" in sf:
                mu_hat = sf["mean"].to_numpy()
            elif "predicted_mean" in sf:
                mu_hat = sf["predicted_mean"].to_numpy()
            else:
                raise RuntimeError("Cannot find 'mean' in summary_frame().")

            ci_lo = (
                sf["mean_ci_lower"].to_numpy()
                if "mean_ci_lower" in sf
                else np.full_like(mu_hat, np.nan, dtype=float)
            )
            ci_hi = (
                sf["mean_ci_upper"].to_numpy()
                if "mean_ci_upper" in sf
                else np.full_like(mu_hat, np.nan, dtype=float)
            )
        else:
            if not hasattr(self._results, "predict"):
                raise RuntimeError("Model results do not support prediction().")
            mu_hat = np.asarray(self._results.predict(x_future)).reshape(-1)
            ci_lo = np.full_like(mu_hat, np.nan, dtype=float)
            ci_hi = np.full_like(mu_hat, np.nan, dtype=float)

        return mu_hat, ci_lo, ci_hi

    def forecast_with_ci(self, horizon: int) -> dict:
        """
        Forecast future values with confidence intervals.

        Parameters
        ----------
        horizon : int
            Forecast horizon.

        Returns
        -------
        dict
            Dictionary with forecast results:
            - mu_hat: point forecasts (mean values)
            - ci_lower: lower confidence interval bounds
            - ci_upper: upper confidence interval bounds
            - pred_lower: lower prediction interval bounds (if supported by subclass)
            - pred_upper: upper prediction interval bounds (if supported by subclass)
            - y_mode: mode of the predictive distribution (if supported by subclass)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._results is None or self._last_train is None:
            raise RuntimeError("Model has not been fitted.")

        mu_hat, ci_lo, ci_hi = self._compute_mean_and_ci(horizon)
        pred_lo, pred_hi, y_mode = self._prediction_intervals(mu_hat)

        return {
            "mu_hat": mu_hat,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "pred_lower": pred_lo,
            "pred_upper": pred_hi,
            "y_mode": y_mode,
        }

    def _prediction_intervals(
        self, mu_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute prediction intervals and distribution modes.

        This is a hook method intended to be overridden by subclasses
        that support distributional forecasts (e.g., Poisson, Negative Binomial).

        Parameters
        ----------
        mu_hat : np.ndarray
            Mean forecasts.

        Returns
        -------
        tuple of np.ndarray
            (pred_lo, pred_hi, y_mode). By default, NaNs are returned.
        """
        n = mu_hat.shape[0]
        na = np.full(n, np.nan, dtype=float)
        return na, na, na

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Forecast mean values.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        series : pd.DataFrame
            Reference time series (used to determine forecast origin).
        covariates : dict, optional
            Additional covariates (not used in the base implementation).

        Returns
        -------
        np.ndarray
            Mean forecast values.
        """
        mu_hat, _, _ = self._compute_mean_and_ci(
            horizon=horizon,
            series=series,
            # covariates=covariates
        )
        return mu_hat

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Draw samples from the predictive distribution.

        Default: fit Normal(mu, sigma) to prediction intervals and sample.
        Subclasses with exact distribution knowledge should override this.
        """
        mu_hat, _, _ = self._compute_mean_and_ci(horizon=horizon, series=series)
        pred_lo, pred_hi, _ = self._prediction_intervals(mu_hat)

        has_intervals = not (np.all(np.isnan(pred_lo)) or np.all(np.isnan(pred_hi)))
        if has_intervals:
            # Recover sigma from (pred_hi - pred_lo) ≈ 2 * z_{1-alpha/2} * sigma
            z = _norm.ppf(1 - self._alpha / 2)
            sigma = np.clip((pred_hi - pred_lo) / (2 * z), 1e-6, None)
        else:
            # No interval info: use 10% of mean as default dispersion
            sigma = np.abs(mu_hat) * 0.1 + 1e-6

        rng = np.random.default_rng()
        return rng.normal(loc=mu_hat, scale=sigma, size=(num_samples, horizon)).astype(
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
        """Return a DistributionForecast fitted to the model's predictive distribution.

        Default: fit Normal from prediction intervals.
        Subclasses override to expose the exact parametric family.
        """
        mu_hat, _, _ = self._compute_mean_and_ci(horizon=horizon, series=series)
        pred_lo, pred_hi, _ = self._prediction_intervals(mu_hat)

        has_intervals = not (np.all(np.isnan(pred_lo)) or np.all(np.isnan(pred_hi)))
        if has_intervals:
            z = _norm.ppf(1 - self._alpha / 2)
            sigma = np.clip((pred_hi - pred_lo) / (2 * z), 1e-6, None)
        else:
            sigma = np.abs(mu_hat) * 0.1 + 1e-6

        return DistributionForecast(
            "normal",
            {
                "loc": mu_hat.astype(np.float32),
                "scale": sigma.astype(np.float32),
            },
        )

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Perform batch forecasting with the model.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: The prediction result.
        """
        raise NotImplementedError("Not implemented batch forecasting!")

    def _init_train_state(self, train_data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """
        Validate univariate input, set training state (_t0, _last_train, _target_col),
        and add a 't' column (days since _t0).

        Parameters
        ----------
        train_data : pd.DataFrame
            Univariate time series with a DatetimeIndex.

        Returns
        -------
        tuple[str, pd.DataFrame]
            (target column name, DataFrame copy with added 't' column).
        """
        target_col = self._check_univariate(train_data)
        self._target_col = target_col
        self._t0 = train_data.index[0]

        train_with_t = train_data.copy()
        train_with_t["t"] = (train_with_t.index - self._t0).days
        self._last_train = train_with_t
        return target_col, train_with_t

    def _build_design_and_target(
        self,
        train_with_t: pd.DataFrame,
        target_col: str,
        add_intercept: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the design matrix from 't' and the target vector.
        If add_intercept=True, the design matrix is [1, t]; otherwise it is [t].

        Parameters
        ----------
        train_with_t : pd.DataFrame
            DataFrame that includes the 't' column.
        target_col : str
            Name of the target column.
        add_intercept : bool, default True
            Whether to prepend an intercept column of ones.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (design_matrix, target).
        """
        time_index_as_days = train_with_t["t"].to_numpy(dtype=float)
        if add_intercept:
            design_matrix = np.column_stack(
                [np.ones_like(time_index_as_days), time_index_as_days]
            )
        else:
            design_matrix = time_index_as_days.reshape(-1, 1)
        target = train_with_t[target_col].to_numpy(dtype=float)
        return design_matrix, target
