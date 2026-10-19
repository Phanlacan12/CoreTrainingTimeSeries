from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from tsfb.base.data_module.preprocess_model.components.irregular_tensors import (
    derive_temporal,
    elapsed_time,
    observation_mask_from_frame,
)
from tsfb.base.data_module.preprocess_model.components.time_mark_builder import (
    build_time_mark_array,
    ensure_datetime_index,
    infer_future_index,
)
from tsfb.base.schema.tfb_family import ModelFamily

DELTA_UNIT = "h"
"""Unit of the raw time gaps fed to the decay models."""

DELTA_SCALE = 48.0
"""Gaps are emitted as ``hours / DELTA_SCALE``.

The decay gates are ``exp(-relu(W d))``, so raw multi-day gaps saturate them to zero and
kill the gradient. Models must be configured with the same ``delta_scale`` to recover
the elapsed time in hours.
"""

MASKED_DECAY_KEYS = ("x", "m", "d", "x_last")


def _as_2d_values(frame: pd.DataFrame, value_cols: List[str]) -> np.ndarray:
    return frame[value_cols].to_numpy(dtype=np.float32, copy=True)


def _masked_decay_arrays(
    frame: pd.DataFrame,
    value_cols: List[str],
    dt_index: pd.DatetimeIndex,
) -> Dict[str, np.ndarray]:
    """Whole-series ``x``/``m``/``d``/``x_last``, before any windowing.

    Derived once over the full frame and only then sliced, so a window inherits the gap
    that preceded it instead of restarting the clock at its own first step. That is safe
    precisely because ``derive_temporal`` is strictly causal: no entry at step ``t``
    depends on a row after ``t``.
    """
    values = _as_2d_values(frame, value_cols)
    mask = observation_mask_from_frame(frame, value_cols)
    observed = np.nan_to_num(values, nan=0.0) * mask
    times = elapsed_time(dt_index, DELTA_UNIT)
    temporal = derive_temporal(
        mask[np.newaxis, ...], times[np.newaxis, ...], observed[np.newaxis, ...]
    )
    return {
        "x": observed,
        "m": mask.astype(np.float32),
        "d": temporal.delta[0] / DELTA_SCALE,
        "x_last": temporal.last_observed[0],
    }


def _empty_samples(
    family: ModelFamily, feature_dim: int, mark_dim: int
) -> Dict[str, Any]:
    if family == ModelFamily.SEQ2SEQ:
        return {
            "x_enc": np.empty((0, 0, feature_dim), dtype=np.float32),
            "x_mark_enc": np.empty((0, 0, mark_dim), dtype=np.float32),
            "x_dec": np.empty((0, 0, feature_dim), dtype=np.float32),
            "x_mark_dec": np.empty((0, 0, mark_dim), dtype=np.float32),
            "y": np.empty((0, 0, feature_dim), dtype=np.float32),
        }
    if family == ModelFamily.MASKED_DECAY:
        empty = np.empty((0, 0, feature_dim), dtype=np.float32)
        keys = MASKED_DECAY_KEYS + ("y", "y_mask")
        return {key: empty.copy() for key in keys}
    return {
        "x": np.empty((0, 0, feature_dim), dtype=np.float32),
        "y": np.empty((0, 0, feature_dim), dtype=np.float32),
    }


# pylint: disable-next=too-many-arguments,too-many-locals
def build_training_samples(
    frame: pd.DataFrame,
    *,
    family: ModelFamily,
    value_cols: List[str],
    seq_len: int,
    pred_len: int,
    label_len: int,
    freq: Any,
    train_ratio: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create train/val window samples from a normalized ungrouped frame."""

    frame = frame.copy()
    dt_index = ensure_datetime_index(frame.index, freq)
    values = _as_2d_values(frame, value_cols)
    marks = build_time_mark_array(dt_index, freq)
    total = len(frame)
    feature_dim = len(value_cols)
    mark_dim = int(marks.shape[1]) if marks.ndim == 2 else 0
    if total < seq_len + pred_len:
        empty = _empty_samples(family, feature_dim, mark_dim)
        return empty, empty

    train_cutoff = (
        total if train_ratio >= 1.0 else max(int(total * train_ratio), seq_len)
    )
    train_samples: Dict[str, List[np.ndarray]] = {}
    val_samples: Dict[str, List[np.ndarray]] = {}
    decay: Dict[str, np.ndarray] = (
        _masked_decay_arrays(frame, value_cols, dt_index)
        if family == ModelFamily.MASKED_DECAY
        else {}
    )

    for start in range(total - seq_len - pred_len + 1):
        enc_slice = slice(start, start + seq_len)
        target_slice = slice(start + seq_len, start + seq_len + pred_len)
        forecast_end = start + seq_len + pred_len
        bucket = train_samples if forecast_end <= train_cutoff else val_samples

        if family == ModelFamily.SEQ2SEQ:
            dec_input = np.concatenate(
                [
                    values[start + seq_len - label_len : start + seq_len],
                    np.zeros((pred_len, feature_dim), dtype=np.float32),
                ],
                axis=0,
            )
            dec_marks = np.concatenate(
                [
                    marks[start + seq_len - label_len : start + seq_len],
                    marks[target_slice],
                ],
                axis=0,
            )
            bucket.setdefault("x_enc", []).append(values[enc_slice])
            bucket.setdefault("x_mark_enc", []).append(marks[enc_slice])
            bucket.setdefault("x_dec", []).append(dec_input)
            bucket.setdefault("x_mark_dec", []).append(dec_marks)
            bucket.setdefault("y", []).append(values[target_slice])
        elif family == ModelFamily.MASKED_DECAY:
            for key in MASKED_DECAY_KEYS:
                bucket.setdefault(key, []).append(decay[key][enc_slice])
            # y is gap-filled with zeros; y_mask tells the loss which steps are real.
            bucket.setdefault("y", []).append(decay["x"][target_slice])
            bucket.setdefault("y_mask", []).append(decay["m"][target_slice])
        else:
            bucket.setdefault("x", []).append(values[enc_slice])
            bucket.setdefault("y", []).append(values[target_slice])

    return _stack_sample_lists(
        train_samples, family, feature_dim, mark_dim
    ), _stack_sample_lists(val_samples, family, feature_dim, mark_dim)


# pylint: disable-next=too-many-arguments,too-many-locals
def build_inference_samples(
    frame: pd.DataFrame,
    *,
    family: ModelFamily,
    value_cols: List[str],
    seq_len: int,
    pred_len: int,
    label_len: int,
    freq: Any,
) -> Dict[str, Any]:
    """Create inference windows from the tail of a normalized frame."""

    frame = frame.copy()
    dt_index = ensure_datetime_index(frame.index, freq)
    if len(frame) < seq_len:
        raise ValueError(
            f"Need at least seq_len={seq_len} rows for inference, got {len(frame)}"
        )
    tail = frame.iloc[-seq_len:]
    tail_index = dt_index[-seq_len:]
    values = _as_2d_values(tail, value_cols)
    marks = build_time_mark_array(tail_index, freq)
    future_index = infer_future_index(tail_index, pred_len, freq)
    future_marks = build_time_mark_array(future_index, freq)

    if family == ModelFamily.SEQ2SEQ:
        dec_input = np.concatenate(
            [
                values[-label_len:],
                np.zeros((pred_len, len(value_cols)), dtype=np.float32),
            ],
            axis=0,
        )
        dec_marks = np.concatenate([marks[-label_len:], future_marks], axis=0)
        return {
            "x_enc": values[np.newaxis, ...],
            "x_mark_enc": marks[np.newaxis, ...],
            "x_dec": dec_input[np.newaxis, ...],
            "x_mark_dec": dec_marks[np.newaxis, ...],
            "y": np.zeros((1, pred_len, len(value_cols)), dtype=np.float32),
            "prediction_index": [future_index],
        }

    if family == ModelFamily.MASKED_DECAY:
        # Derived over the whole frame, then tailed: the first step of the window keeps
        # the gap that preceded it rather than starting from a clean slate.
        decay = _masked_decay_arrays(frame, value_cols, dt_index)
        samples: Dict[str, Any] = {
            key: decay[key][-seq_len:][np.newaxis, ...] for key in MASKED_DECAY_KEYS
        }
        samples["y"] = np.zeros((1, pred_len, len(value_cols)), dtype=np.float32)
        samples["y_mask"] = np.ones((1, pred_len, len(value_cols)), dtype=np.float32)
        samples["prediction_index"] = [future_index]
        return samples

    return {
        "x": values[np.newaxis, ...],
        "y": np.zeros((1, pred_len, len(value_cols)), dtype=np.float32),
        "prediction_index": [future_index],
    }


def _stack_sample_lists(
    samples: Dict[str, List[np.ndarray]],
    family: ModelFamily,
    feature_dim: int,
    mark_dim: int,
) -> Dict[str, Any]:
    if not samples:
        return _empty_samples(family, feature_dim, mark_dim)
    return {key: np.stack(value).astype(np.float32) for key, value in samples.items()}
