import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore
from torch.utils.data import DataLoader

from tsfb.base.utils.data_processing import split_time
from tsfb.base.utils.timefeatures import time_features


def train_val_split(train_data, ratio, seq_len):
    """
    Split the input time series data into training and validation sets.

    Args:
        train_data (pd.DataFrame): The input time series data.
        ratio (float): The ratio of data to use for training (between 0 and 1).
        seq_len (int or None): The sequence length
            for splitting. If None, no sequence overlap is considered.

    Returns:
        tuple: (train_data_value, valid_data) where valid_data
            may be None if ratio == 1.
    """
    if ratio == 1:
        return train_data, None

    if seq_len is not None:
        border = int((train_data.shape[0]) * ratio)
        train_data_value, valid_data_rest = split_time(train_data, border)
        _, valid_data = split_time(train_data, border - seq_len)
        return train_data_value, valid_data

    border = int((train_data.shape[0]) * ratio)
    train_data_value, valid_data_rest = split_time(train_data, border)
    return train_data_value, valid_data_rest


def decompose_time(
    time: np.ndarray,
    freq: str,
) -> np.ndarray:
    """
    Split the given array of timestamps into components based on the frequency.

    :param time: Array of timestamps.
    :param freq: The frequency of the time stamp.
    :return: Array of timestamp components.
    """
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {
        "m": 0,
        "w": 1,
        "b": 2,
        "d": 2,
        "h": 3,
        "t": 4,
        "s": 5,
    }
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(
    time_stamp: np.ndarray,
    timeenc: int,
    freq: str,
) -> np.ndarray:
    """
    Extract temporal features from the time stamp.

    :param time_stamp: The time stamp ndarray.
    :param timeenc: The time encoding type.
    :param freq: The frequency of the time stamp.
    :return: The mark of the time stamp.
    """
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError(f"Unknown time encoding {timeenc}")
    return data_stamp.astype(np.float32)


def forecasting_data_provider(  # pylint: disable=too-many-arguments
    data, config, timeenc, batch_size, shuffle, drop_last
):
    """
    Build a PyTorch Dataset and DataLoader for transformer-based forecasting.

    This function wraps the input time series into a `DatasetForTransformer`
    instance and returns both the dataset and a `DataLoader` for efficient
    batching during training or evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with a datetime index and numeric columns.
    config : object
        Configuration object with at least the following attributes:
            - seq_len : int
                Length of historical input sequence.
            - pred_len : int
                Forecast horizon length.
            - label_len : int
                Number of past values fed into the decoder.
            - freq : str
                Frequency string (e.g., "h", "d").
            - num_workers : int
                Number of workers for DataLoader.
    timeenc : int
        Encoding type flag for time features (passed to `get_time_mark`).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset during iteration.
    drop_last : bool
        Whether to drop the last incomplete batch.

    Returns
    -------
    tuple
        dataset : DatasetForTransformer
            The constructed dataset object.
        data_loader : torch.utils.data.DataLoader
            DataLoader wrapping the dataset with given batch size and options.
    """
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.pred_len,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )

    return dataset, data_loader


class DatasetForTransformer:
    """
    A custom PyTorch-style dataset for preparing sequential data to train
    and evaluate transformer-based forecasting models.

    This dataset slices a time series DataFrame into overlapping windows
    consisting of a history (input), label (decoder input), and prediction horizon.
    Additionally, it generates corresponding time encoding features (marks).

    Attributes
    ----------
    dataset : pd.DataFrame
        The full time series data with a datetime index and numeric values.
    history_length : int
        Number of time steps to use as historical input (encoder length).
    prediction_length : int
        Number of time steps to forecast (output horizon).
    label_length : int
        Number of time steps used as decoder input.
    current_index : int
        Tracks current position in dataset (not usually used externally).
    timeenc : int
        Encoding type flag for time features (passed to `get_time_mark`).
    freq : str
        Frequency string (e.g., 'h' for hourly, 'd' for daily).
    data_stamp : np.ndarray
        Encoded time features aligned with the dataset index.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
    ):
        """
        Initialize the DatasetForTransformer.

        Parameters
        ----------
        dataset : pd.DataFrame
            Time series data with a datetime index and numeric values.
        history_len : int, optional, default=10
            Number of steps used for the encoder input (lookback window).
        prediction_len : int, optional, default=2
            Number of steps to predict (forecast horizon).
        label_len : int, optional, default=5
            Number of steps fed into the decoder as known past values.
        timeenc : int, optional, default=1
            Flag for selecting time encoding method.
        freq : str, optional, default="h"
            Frequency string for generating time features ('h' = hourly).
        """
        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __len__(self) -> int:
        """
        Return the total number of samples that can be drawn from the dataset.

        Each sample is defined as one rolling window of history, label,
        and prediction slices.

        Returns
        -------
        int
            The number of available samples.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __read_data__(self):
        """
        Precompute time encoding features ("marks") for the dataset.
        """
        idx = self.dataset.index

        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception as e:
                raise ValueError(
                    "Dataset index must be datetime-like or convertible to datetime. "
                    f"Current index dtype: {self.dataset.index.dtype}"
                ) from e

        time_arr = idx.to_numpy().reshape(1, -1)

        data_stamp = get_time_mark(time_arr, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Retrieve a single sample consisting of input, output, and time marks.

        Parameters
        ----------
        index : int
            The starting position in the dataset for slicing.

        Returns
        -------
        tuple of torch.Tensor
            - seq_x : torch.FloatTensor
                Historical input sequence of shape (history_len, n_features).
            - seq_y : torch.FloatTensor
                Target sequence (label_len + prediction_len, n_features).
            - seq_x_mark : torch.FloatTensor
                Time encoding for the historical input sequence.
            - seq_y_mark : torch.FloatTensor
                Time encoding for the target sequence.
        """
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


def load_config(config_path: str) -> dict:
    """
    Load a configuration file from a YAML path.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
