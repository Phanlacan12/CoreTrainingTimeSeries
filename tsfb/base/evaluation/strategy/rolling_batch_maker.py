from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class RollingForecastEvalBatchMaker:
    """Class for creating evaluation batches in rolling forecast strategy.

    This class handles creating batches of data for training and evaluation
    in a rolling window fashion.

    Attributes:
        series (pd.DataFrame): The input time series data
        index (pd.Index): Index of the time series
        _series_np (np.ndarray): NumPy array of series values
        index_list (List[int]): List of indices for rolling windows
        current_sample_count (int): Current number of processed samples
        covariates (Dict[str, pd.DataFrame]): Dictionary of covariate data
    """

    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Initialize the batch maker.

        Args:
            series: Input time series data
            index_list: List of indices for rolling windows
            covariates: Optional dictionary of covariate data
        """
        self.series = series
        self.index = series.index
        self._series_np = series.values
        self.index_list = index_list
        self.current_sample_count = 0
        self.covariates = covariates or {}

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        """Create a batch of data for prediction.

        Args:
            batch_size: Size of the batch
            win_size: Size of the rolling window

        Returns:
            Dictionary containing input batch and aligned covariates
        """
        idxs = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        self.current_sample_count += len(idxs)

        predict_batch = self._make_batch_data(
            self._series_np, np.array(idxs) - win_size, win_size
        )

        # covariates batch
        cov_batch = {}
        for cov_type, df in self.covariates.items():
            arr = df.values
            cov_batch[cov_type] = self._make_batch_data(
                arr, np.array(idxs) - win_size, win_size
            )

        return {
            "input": predict_batch,
            "covariates": cov_batch,
            "input_index": [self.index[i - win_size : i] for i in idxs],
        }

    def make_batch_eval(self, horizon: int) -> dict:
        """Create a batch of data for evaluation.

        Args:
            horizon: Forecast horizon length

        Returns:
            Dictionary containing target data and covariates for evaluation
        """
        targets = self._make_batch_data(
            self._series_np, np.array(self.index_list), horizon
        )
        cov_batch = {}
        for cov_type, df in self.covariates.items():
            arr = df.values
            cov_batch[cov_type] = self._make_batch_data(
                arr, np.array(self.index_list), horizon
            )

        return {
            "target": targets,
            "covariates": cov_batch,
        }

    def has_more_batches(self) -> bool:
        """Check if more batches are available.

        Returns:
            True if more batches can be created, False otherwise
        """
        return self.current_sample_count < len(self.index_list)

    @staticmethod
    def _make_batch_data(
        data: Any, index_list: np.ndarray, win_size: int
    ) -> np.ndarray:
        """Create batch data using efficient numpy indexing.

        Args:
            data: Input data array
            index_list: List of indices
            win_size: Size of the window

        Returns:
            NumPy array containing batch data
        """
        windows = sliding_window_view(data, window_shape=(win_size, *data.shape[1:]))
        return np.squeeze(windows[index_list])


class RollingForecastPredictBatchMaker:
    """Wrapper class for making prediction batches.

    This class provides an interface for creating prediction batches
    using a RollingForecastEvalBatchMaker instance.
    """

    def __init__(self, batch_maker: RollingForecastEvalBatchMaker):
        """Initialize with a batch maker instance.

        Args:
            batch_maker: RollingForecastEvalBatchMaker instance
        """
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """Create a batch for prediction.

        Args:
            batch_size: Size of the batch
            win_size: Size of the rolling window

        Returns:
            Dictionary containing batch data
        """
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        """Check if more batches are available.

        Returns:
            True if more batches can be created, False otherwise
        """
        return self._batch_maker.has_more_batches()
