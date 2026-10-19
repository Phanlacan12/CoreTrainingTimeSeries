import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries

from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class DartsConfig:
    """
    Configuration handler for Darts models.

    Stores and manages model hyperparameters, provides convenient access methods,
    and ensures safe single-GPU usage for Darts models.
    """

    def __init__(self, **kwargs):
        """
        Initialize DartsConfig with model hyperparameters.
        :param kwargs: Model hyperparameters as keyword arguments.
        """
        self.params = {
            **kwargs,
        }

    def __getattr__(self, key: str) -> Any:
        """
        Allow attribute-style access to parameters.
        :param key: Parameter name.
        :return: Parameter value.
        """
        return self.get(key)

    def __getitem__(self, key: str) -> Any:
        """
        Allow dict-style access to parameters.
        :param key: Parameter name.
        :return: Parameter value.
        """
        return self.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value with an optional default.
        :param key: Parameter name.
        :param default: Default value if key is not found.
        :return: Parameter value or default.
        """
        params = self.__dict__.get("params", None)
        if params is None:
            return default
        return params.get(key, default)

    def get_darts_class_params(self) -> dict:
        """
        Get parameters for Darts model class,
        removing normalization and fixing GPU settings.
        :return: Dictionary of parameters for Darts model class.
        """
        ret = self.params.copy()
        # ret.pop("norm")
        self._fix_multi_gpu(ret)

        if any(k in ret for k in ["P_seasonal", "D_seasonal", "Q_seasonal"]):
            seasonal_p = ret.pop("P_seasonal", 0)
            seasonal_d = ret.pop("D_seasonal", 0)
            seasonal_q = ret.pop("Q_seasonal", 0)
            seasonal_periods = ret.pop("seasonal_periods", 12)
            ret["seasonal_order"] = (
                seasonal_p,
                seasonal_d,
                seasonal_q,
                seasonal_periods,
            )

        return ret

    def _fix_multi_gpu(self, args_dict: Dict) -> None:
        """
        Check and disable using multi-gpu per task.
            Ensures only one GPU is used for Darts models.
        :param args_dict: Argument dictionary to be passed to Darts models.
        """
        # CUDA_VISIBLE_DEVICES should be set by the parallel backend
        gpu_devices = list(
            filter(None, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        )
        if len(gpu_devices) > 1:
            pl_args = args_dict.get("pl_trainer_kwargs", {})
            device_args = pl_args.get("devices", None)
            if (
                device_args is None
                or (isinstance(device_args, list) and len(device_args) > 1)
                or (isinstance(device_args, int) and device_args > 1)
            ):
                args_dict.setdefault("pl_trainer_kwargs", {})
                args_dict["pl_trainer_kwargs"]["devices"] = [0]
                logger.warning(
                    "Multi-gpu training is not supported, using only gpu %s",
                    gpu_devices[0],
                )


@dataclass
class FactoryConfig:
    """
    Configuration for creating a forecasting model from a factory.

    Attributes:
        model_class (type): The class of the model to be instantiated.
        model_args (dict): Arguments used for model initialization.
        model_name (str): Name of the model.
        required_args (dict): Dictionary of required arguments for the model.
        allow_fit_on_eval (bool): Whether fitting is allowed on evaluation data.
        supports_validation (bool): Whether the model
            supports validation during training.
    """

    model_class: type
    model_args: dict
    model_name: str
    required_args: dict
    allow_fit_on_eval: bool
    supports_validation: bool


@dataclass
class CovariateWindowConfig:
    """
    Configuration for extracting a covariate window for forecasting.

    Attributes:
        cov_dict (Dict[str, np.ndarray]): Dictionary containing covariate arrays.
        window_idx (pd.Index): Time index of the current window.
        window_index (int): Index of the current window in the sequence.
        lookback (int): Number of timesteps to look back.
        output_len (int): Length of forecast output.
    """

    cov_dict: Dict[str, np.ndarray]
    window_idx: pd.Index
    window_index: int
    lookback: int
    output_len: int


@dataclass
class CovariateSeries:
    """
    TimeSeries objects representing past and future covariates.

    Attributes:
        past_train (Optional[TimeSeries]): Past covariates for training.
        past_val (Optional[TimeSeries]): Past covariates for validation.
        future_train (Optional[TimeSeries]): Future covariates for training.
        future_val (Optional[TimeSeries]): Future covariates for validation.
    """

    past_train: Optional[TimeSeries] = None
    past_val: Optional[TimeSeries] = None
    future_train: Optional[TimeSeries] = None
    future_val: Optional[TimeSeries] = None


@dataclass
class TimeSeriesGroup:
    """
    A grouped object representing the training and validation series with covariates.

    Attributes:
        ts_train (TimeSeries): Target time series for training.
        ts_valid (Optional[TimeSeries]): Target time series for validation.
        covariates (CovariateSeries): Covariates associated with the target series.
    """

    ts_train: TimeSeries
    ts_valid: Optional[TimeSeries]
    covariates: CovariateSeries


@dataclass
class ForecastWindowConfig:
    """
    Configuration for forecasting a single sliding window.

    Attributes:
        i (int): Index of the current window.
        start (int): Start position of the window in the full index.
        full_idx (pd.Index): Full time index of the data.
        x_window (np.ndarray): Input window data for prediction.
        cov_dict (Dict[str, np.ndarray]): Covariate arrays for the window.
        lookback (int): Lookback length for the input window.
        output_len (int): Forecast horizon.
        win_size (int): Total window size (lookback + output).
    """

    i: int
    start: int
    full_idx: pd.Index
    x_window: np.ndarray
    cov_dict: Dict[str, np.ndarray]
    lookback: int
    output_len: int
    win_size: int


@dataclass
class AllWindowsForecastConfig:
    """
    Configuration for forecasting across multiple sliding windows.

    Attributes:
        eval_maker (Any): Object responsible for generating evaluation window indices.
        x_windows (np.ndarray): Input windows array of shape (n_windows, ...).
        cov_dict (Dict[str, np.ndarray]): Dictionary of covariate arrays.
        full_idx (pd.Index): Full time index of the data.
        lookback (int): Lookback length for each input window.
        output_len (int): Forecast horizon.
    """

    eval_maker: Any
    x_windows: np.ndarray
    cov_dict: Dict[str, np.ndarray]
    full_idx: pd.Index
    lookback: int
    output_len: int
