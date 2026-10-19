from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tsfb.base.utils.timefeatures import time_features


def _freq_to_pandas(freq: Any) -> str:
    if not freq:
        return "D"
    freq_str = str(freq)
    mapping = {
        "h": "H",
        "d": "D",
        "w": "W",
        "m": "M",
        "t": "min",
        "min": "min",
    }
    return mapping.get(freq_str.lower(), freq_str)


def ensure_datetime_index(index: pd.Index, freq: Any) -> pd.DatetimeIndex:
    """Coerce an index to DatetimeIndex, creating a synthetic one if needed."""

    if isinstance(index, pd.DatetimeIndex):
        return index
    periods = len(index)
    return pd.date_range("2000-01-01", periods=periods, freq=_freq_to_pandas(freq))


def build_time_mark_array(index: pd.Index, freq: Any) -> np.ndarray:
    """Build the model time-feature matrix aligned to an index."""
    dt_index = ensure_datetime_index(index, freq)
    feats = time_features(dt_index, freq=str(freq or "d"))
    return feats.transpose(1, 0).astype(np.float32)


def infer_future_index(index: pd.Index, horizon: int, freq: Any) -> pd.DatetimeIndex:
    """Infer the next `horizon` timestamps after the provided index."""
    dt_index = ensure_datetime_index(index, freq)
    pandas_freq = _freq_to_pandas(freq)
    return pd.date_range(dt_index[-1], periods=horizon + 1, freq=pandas_freq)[1:]
