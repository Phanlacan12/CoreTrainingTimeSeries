from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.base_model import BatchMaker
from tsfb.base.models.base_statistic import BaseStatistic


class PoissonGLM(BaseStatistic):
    """
    Poisson GLM for univariate time series forecasting.

    Assumes target counts follow a Poisson distribution with log link function.
    """

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "PoissonGLM":
        target_col, train_with_t = self._init_train_state(train_data)
        design_matrix, target = self._build_design_and_target(
            train_with_t, target_col, add_intercept=True
        )

        model = sm.GLM(target, design_matrix, family=sm.families.Poisson())
        self._results = model.fit()
        return self

    def _prediction_intervals(
        self, mu_hat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate prediction intervals using quantiles of Poisson distribution.

        Parameters
        ----------
        mu_hat : np.ndarray
            Mean forecasts (λ)

        Returns
        -------
        pred_lo : np.ndarray
            Lower prediction bound (e.g., 2.5% quantile)
        pred_hi : np.ndarray
            Upper prediction bound (e.g., 97.5% quantile)
        y_mode : np.ndarray
            Mode of Poisson (floor of mean)
        """

        alpha = self._alpha
        pred_lo = poisson.ppf(alpha / 2, mu_hat)
        pred_hi = poisson.ppf(1 - alpha / 2, mu_hat)
        y_mode = np.floor(mu_hat)

        return pred_lo, pred_hi, y_mode

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Draw samples from the Poisson predictive distribution (exact)."""
        mu_hat, _, _ = self._compute_mean_and_ci(horizon=horizon, series=series)
        mu_clipped = np.clip(mu_hat, 1e-6, None)
        rng = np.random.default_rng()
        return rng.poisson(lam=mu_clipped, size=(num_samples, horizon)).astype(
            np.float32
        )

    def forecast_distribution(  # pylint: disable=too-many-arguments,unused-argument
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "poisson",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> DistributionForecast:
        """Return exact Poisson(mu) distribution fitted by the GLM."""
        mu_hat, _, _ = self._compute_mean_and_ci(horizon=horizon, series=series)
        return DistributionForecast(
            "poisson", {"mu": np.clip(mu_hat, 1e-6, None).astype(np.float32)}
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
