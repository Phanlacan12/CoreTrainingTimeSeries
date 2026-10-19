# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from tsfb.base.evaluation.strategy.constants import (
    DEFAULT_DISTRIBUTION_TYPE,
    DEFAULT_PROBABILISTIC_NUM_SAMPLES,
    DEFAULT_PROBABILISTIC_OUTPUT_FORMAT,
    DEFAULT_QUANTILE_LEVELS,
)


@dataclass
class ForecastConfig:
    """Configuration for forecasting.

    Attributes:
        horizon (int): The forecast horizon (number of future time steps to predict)
        save_tp (bool): Whether to save true and predicted values
        train_ratio (float): Ratio of training data in train+validation split
        train_len (int): Length of training data
        is_multi (bool): Whether this is a multivariate forecast
        return_mps (bool): Whether to return per-series metrics
    """

    horizon: int
    save_tp: bool
    train_ratio: float
    train_len: int
    calib_len: int
    prediction_start: int
    output_shift: int
    is_multi: bool
    return_mps: bool = False
    use_conformal: bool = False
    quantile_levels: List[float] = None  # type: ignore[assignment]
    probabilistic_num_samples: int = DEFAULT_PROBABILISTIC_NUM_SAMPLES
    probabilistic_output_format: str = DEFAULT_PROBABILISTIC_OUTPUT_FORMAT
    distribution_type: str = DEFAULT_DISTRIBUTION_TYPE

    def __post_init__(self) -> None:
        if self.quantile_levels is None:
            self.quantile_levels = list(DEFAULT_QUANTILE_LEVELS)


@dataclass
class CovariateData:
    """Container for covariate data.

    Holds training and prediction covariates data.

    Attributes:
        train (Dict[str, pd.DataFrame]): Covariates for training
        pred (Dict[str, pd.DataFrame]): Covariates for prediction
    """

    train: Dict[str, pd.DataFrame]
    pred: Dict[str, pd.DataFrame]

    @classmethod
    def from_raw_data(
        cls,
        covariates: Optional[Dict[str, pd.DataFrame]],
        train_len: int,
        horizon: int,
        output_shift: int = 0,
    ) -> "CovariateData":
        """Create CovariateData from raw covariates dictionary.

        Args:
            covariates: Dictionary of covariate DataFrames by type
            train_len: Length of training data for splitting past covariates

        Returns:
            CovariateData instance with processed covariates
        """
        cov_train: Dict[str, pd.DataFrame] = {}
        cov_pred: Dict[str, pd.DataFrame] = {}

        if covariates:
            raw_past = covariates.get("past_covariates")
            raw_future = covariates.get("future_covariates")
            raw_static = covariates.get("static_covariates")

            if raw_past is not None:
                cov_train["past_covariates"] = raw_past.iloc[:train_len]
                end = min(
                    len(raw_past),
                    train_len + int(horizon) + max(int(output_shift), 0),
                )
                cov_pred["past_covariates"] = raw_past.iloc[:end]

            if raw_future is not None:
                cov_train["future_covariates"] = raw_future
                cov_pred["future_covariates"] = raw_future

            if raw_static is not None:
                cov_train["static_covariates"] = raw_static
                cov_pred["static_covariates"] = raw_static

        return cls(train=cov_train, pred=cov_pred)


@dataclass
class ModelOutput:
    """Container for forecast outputs and timing metadata."""

    predictions: pd.DataFrame
    fit_time: float
    predict_time: float
