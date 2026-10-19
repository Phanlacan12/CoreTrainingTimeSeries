from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.models.base_model import BatchMaker


class ForecastingApproach(ABC):
    """
    The base class for forecasting approaches.
    """

    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None

    @abstractmethod
    def forecast_fit(self, train_data: pd.DataFrame, **kwargs) -> "ForecastingApproach":
        """
        Fit the forecasting model on the training data.

        :param train_data: The training data as a pandas DataFrame.
        :return: An instance of the fitted forecasting approach.
        """

    @abstractmethod
    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict future values based on the test data.

        :return: A DataFrame containing the predicted values.
        """

    @abstractmethod
    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict future values for *all* remaining windows provided by a BatchMaker.

        :param horizon: Number of time steps to predict for each window.
        :param batch_maker: An instance of BatchMaker
            (e.g. RollingForecastPredictBatchMaker)
            that yields batches of look-back windows.
        :param covariates: (Optional) any global covariates dict to pass down.
        :return: numpy array of shape (n_windows, horizon) or
                 (n_windows, horizon, n_features).
        """

    @staticmethod
    def _safe_infer_freq(index: pd.Index) -> str:
        """
        Infer a safe frequency string even if index is non-datetime.
        Handles DatetimeIndex, RangeIndex, Int64Index gracefully.
        Fallback: 'H' (hourly).
        """
        if isinstance(index, pd.DatetimeIndex):
            freq = pd.infer_freq(index)
            if freq is not None:
                return freq
            return "H"
        if isinstance(index, pd.RangeIndex):
            step = index.step or 1
            # interpret step as hours if small, else days
            return "H" if step <= 24 else "D"
        if np.issubdtype(index.dtype, np.number):
            diffs = np.diff(index[: min(5, len(index))])
            step = diffs[0] if len(diffs) > 0 else 1
            return "H" if step <= 24 else "D"
        return "H"

    @staticmethod
    def _safe_step(index: pd.Index) -> int:
        if isinstance(index, pd.RangeIndex):
            return index.step or 1
        if np.issubdtype(index.dtype, np.number):
            diffs = np.diff(index[: min(5, len(index))])
            return int(diffs[0]) if len(diffs) > 0 else 1
        return 1


class GroupedForecastingApproach(ForecastingApproach):
    """Shared grouped-mode helpers for forecasting approaches."""

    def __init__(self) -> None:
        super().__init__()
        self.grouped_mode = False
        self.group_col: Optional[str] = None
        self.time_col: Optional[str] = None
        self.target_cols: List[str] = []

    @staticmethod
    def _resolve_group_meta(
        frame: pd.DataFrame,
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str], List[str]]:
        group_col = kwargs.get("group_col") or next(
            iter(kwargs.get("group_cols", []) or []),
            None,
        )
        time_col = kwargs.get("time_col")
        if (
            group_col is None
            or time_col is None
            or group_col not in frame.columns
            or time_col not in frame.columns
        ):
            return False, None, None, []

        target_cols = list(kwargs.get("target_cols") or [])
        if not target_cols:
            target_cols = [
                col
                for col in frame.columns
                if col not in {group_col, time_col}
                and pd.api.types.is_numeric_dtype(frame[col])
            ]
        return True, group_col, time_col, target_cols

    @staticmethod
    def _sort_grouped_frame(
        frame: pd.DataFrame,
        group_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        result[time_col] = pd.to_datetime(result[time_col])
        return result.sort_values([group_col, time_col]).reset_index(drop=True)

    def _configure_grouped_mode(
        self,
        frame: pd.DataFrame,
        kwargs: Dict[str, Any],
    ) -> bool:
        (
            self.grouped_mode,
            self.group_col,
            self.time_col,
            self.target_cols,
        ) = self._resolve_group_meta(frame, kwargs)
        return self.grouped_mode

    def _maybe_forecast_grouped(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        return (
            self._forecast_grouped(
                horizon,
                lookback_data,
                covariates=covariates,
                **kwargs,
            )
            if self.grouped_mode
            else None
        )

    def _finalize_grouped_outputs(self, outputs: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate grouped forecasts and restore grouped ordering."""
        assert self.group_col is not None
        assert self.time_col is not None
        return self._sort_grouped_frame(
            pd.concat(outputs, ignore_index=True),
            self.group_col,
            self.time_col,
        )

    @abstractmethod
    def _forecast_grouped(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Forecast grouped data for approaches that support grouped mode."""
