from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.forecasting.probabilistic_forecast import ProbabilisticForecast
from tsfb.base.models.base_model import BatchMaker, ModelBase


class DefaultApproach(ForecastingApproach):
    """A default forecasting approach that wraps a model
    factory for time series forecasting.

    This class implements the `ForecastingApproach` interface, providing
    methods to fit a model
    and generate forecasts or batch forecasts using a provided model factory.
    The model is initialized during fitting and used for
    subsequent forecasting tasks.

    Attributes:
        model_factory (Callable[[], ModelBase]): A callable that returns a new
            instance of a `ModelBase`-derived model.
        model (Optional[ModelBase]): The instantiated model,
            set after calling `forecast_fit`.
    """

    def __init__(self, model_factory: Callable[[], ModelBase]):
        """Initialize the DefaultApproach with a model factory.

        Args:
            model_factory (Callable[[], ModelBase]): A callable that
                creates a new instance of
                a `ModelBase`-derived model when invoked.
        """
        super().__init__()
        self.model_factory = model_factory
        self.model: Optional[ModelBase] = None

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ):
        """Fit the forecasting model using the provided training data.

        Instantiates a new model using the model factory and
        fits it with the training data and optional covariates.

        Args:
            train_data (pd.DataFrame): The training data as a pandas
                DataFrame, typically
                containing time series data with a DatetimeIndex.
            covariates (Optional[Dict[str, pd.DataFrame]]):
                Optional dictionary of covariate
                DataFrames, where keys are covariate names and values
                are DataFrames with matching indices. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model's
                `forecast_fit` method.
        """
        self.model = self.model_factory()
        self.model.forecast_fit(train_data, covariates=covariates, **kwargs)

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate forecasts for the specified horizon using lookback data.

        Produces forecasts using the fitted model. The output is a pandas DataFrame with
        a DatetimeIndex, ensuring compatibility with time series data.

        Args:
            horizon (int): The number of time steps to forecast into the future.
            lookback_data (pd.DataFrame): Historical data used as input for forecasting,
                typically containing time series data with a DatetimeIndex.
            covariates (Optional[Dict[str, pd.DataFrame]]): Optional dictionary
                of covariate DataFrames for the forecast period. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying model's
                `forecast` method.

        Returns:
            pd.DataFrame: A DataFrame containing the forecasted values,
                with a DatetimeIndex and columns matching `lookback_data`.

        Raises:
            RuntimeError: If the model is not trained
                (i.e., `forecast_fit` has not been called).
            TypeError: If the model's forecast output is of an unsupported type.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call forecast_fit() first.")

        pred = self.model.forecast(
            horizon, lookback_data, covariates=covariates, **kwargs
        )

        group_col = kwargs.get("group_col")
        time_col = kwargs.get("time_col")

        if isinstance(pred, np.ndarray):
            pred = pd.DataFrame(pred, columns=lookback_data.columns)
        elif isinstance(pred, pd.Series):
            pred = pred.to_frame()
        elif not isinstance(pred, pd.DataFrame):
            raise TypeError(f"Unsupported prediction type: {type(pred)}")

        assert isinstance(pred, pd.DataFrame)

        if (group_col and group_col in pred.columns) or (
            time_col and time_col in pred.columns
        ):
            return pred

        if pred.index is None or not isinstance(pred.index, pd.DatetimeIndex):
            last_index = lookback_data.index[-1]
            freq = self._safe_infer_freq(lookback_data.index)
            if isinstance(lookback_data.index, pd.DatetimeIndex):
                pred.index = pd.date_range(
                    start=last_index + pd.Timedelta(1, freq),
                    periods=horizon,
                    freq=freq,
                )
            else:
                step = self._safe_step(lookback_data.index)
                start_val = last_index + step
                pred.index = pd.RangeIndex(
                    start=start_val, stop=start_val + horizon * step, step=step
                )

        return pred

    def forecast_probabilistic(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        num_samples: int = 100,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Optional[ProbabilisticForecast]:
        """Return a ProbabilisticForecast if the underlying model supports it.

        Delegates to model.forecast_probabilistic() when supports_probabilistic=True.
        Returns None for models that only support point forecasts.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call forecast_fit() first.")
        if not getattr(self.model, "supports_probabilistic", False):
            return None
        samples = self.model.forecast_probabilistic(  # type: ignore[attr-defined]
            horizon,
            lookback_data,
            num_samples=num_samples,
            covariates=covariates,
            **kwargs,
        )
        return ProbabilisticForecast(samples)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Optional[DistributionForecast]:
        """Return a DistributionForecast if the underlying model supports it.

        Delegates to model.forecast_distribution() when supports_probabilistic=True.
        The default implementation fits a Normal to samples via moment-matching.
        Returns None for models that only support point forecasts.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call forecast_fit() first.")
        if not getattr(self.model, "supports_probabilistic", False):
            return None
        return self.model.forecast_distribution(  # type: ignore[attr-defined]
            horizon,
            lookback_data,
            distribution_name=distribution_name,
            num_samples=num_samples,
            covariates=covariates,
            **kwargs,
        )

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Delegate batch forecasting to the underlying model.

        Args:
            horizon (int): The number of time steps to forecast for each window.
            batch_maker (BatchMaker): Provides batched look-back windows.
            covariates (Optional[Dict[str, Any]]): Passed through to the model.
            **kwargs: Additional keyword arguments for the model's batch_forecast.

        Returns:
            np.ndarray of shape (n_windows, horizon) or
            (n_windows, horizon, n_features).

        Raises:
            RuntimeError: If the model is not trained.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call forecast_fit() first.")
        return self.model.batch_forecast(horizon, batch_maker, **kwargs)
