"""Observation mask and time-gap tensors for irregularly sampled series.

These are the leakage-critical primitives behind the decay-based models
(GRU-D, BRITS, T-LSTM). They deliberately live here rather than in
``origin_code/`` so they are linted, typed and covered by the project's own tests:
``last_observed`` being strictly causal is a correctness guarantee, not vendored code.

Ported from the ISMTS-Review benchmark harness
(``survey_irregular_0/ISMTS-Review/benchmark/data/windowing.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalTensors:
    """Time-gap tensors derived from an observation mask.

    Returned as a dataclass rather than a tuple so a future field cannot silently
    shift the positional meaning of the existing ones.
    """

    delta: np.ndarray
    """(N, T, D) time since the feature was last seen, in the unit of ``times``."""

    last_observed: np.ndarray
    """(N, T, D) last observed value, STRICTLY before the current step."""

    last_valid: np.ndarray
    """(N, T, D) uint8, 0 until a feature's first observation."""


def derive_temporal(
    mask: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
) -> TemporalTensors:
    """Che et al. (2018) time gaps and last-observed carry-forward.

        delta[:, 0, :] = 0
        delta[:, t, :] = (times[t] - times[t-1]) + (1 - mask[t-1]) * delta[t-1]

    Both are computed from whatever mask the caller passes; callers pass the
    *visible* mask (observed minus artificially eliminated), never the raw one.

    ``last_observed`` is STRICTLY causal: at step t it holds the last observation
    before t, never the one at t. That matches ``delta`` (which measures the gap to
    the previous observation) and, crucially, means a model's decay estimate at an
    observed step can be scored against that step's true value without trivially
    reproducing it.

    Args:
        mask: (N, T, D) 1 where the feature is observed.
        times: (N, T) timestamps of each step, in a consistent unit.
        values: (N, T, D) observed values; entries where ``mask`` is 0 are ignored.

    Returns:
        TemporalTensors with ``delta``, ``last_observed`` and ``last_valid``.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be (N, T, D), got shape {mask.shape}")
    if times.shape != mask.shape[:2]:
        raise ValueError(
            f"times must be (N, T)={mask.shape[:2]}, got shape {times.shape}"
        )
    if values.shape != mask.shape:
        raise ValueError(
            f"values must match mask shape {mask.shape}, got {values.shape}"
        )

    n_series, n_steps, n_features = mask.shape
    gaps = np.zeros((n_series, n_steps), dtype=np.float32)
    gaps[:, 1:] = np.diff(times, axis=1)
    gaps = np.maximum(gaps, 0.0)

    delta = np.zeros((n_series, n_steps, n_features), dtype=np.float32)
    last = np.zeros((n_series, n_steps, n_features), dtype=np.float32)
    last_valid = np.zeros((n_series, n_steps, n_features), dtype=np.uint8)

    for step in range(1, n_steps):
        prev_mask = mask[:, step - 1, :].astype(np.float32)
        delta[:, step, :] = (
            gaps[:, step, None] + (1.0 - prev_mask) * delta[:, step - 1, :]
        )

    prev_val = np.zeros((n_series, n_features), dtype=np.float32)
    prev_seen = np.zeros((n_series, n_features), dtype=bool)
    for step in range(n_steps):
        last[:, step, :] = prev_val
        last_valid[:, step, :] = prev_seen
        seen_now = mask[:, step, :].astype(bool)
        prev_val = np.where(seen_now, values[:, step, :], prev_val)
        prev_seen = prev_seen | seen_now

    return TemporalTensors(delta=delta, last_observed=last, last_valid=last_valid)


def forward_fill(
    values: np.ndarray,
    mask: np.ndarray,
    fill_value: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Dense series: carry the last observation forward, then fall back to a constant.

    ``fill_value`` is broadcast over (D,) -- normally the train-fold feature mean,
    which is 0 after z-scoring, so this reduces to mean-imputation before a feature's
    first observation.
    """
    n_series, n_steps, n_features = values.shape
    out = np.empty_like(values, dtype=np.float32)
    current = np.broadcast_to(
        np.asarray(fill_value, dtype=np.float32), (n_series, n_features)
    ).copy()
    for step in range(n_steps):
        seen_now = mask[:, step, :].astype(bool)
        current = np.where(seen_now, values[:, step, :], current)
        out[:, step, :] = current
    return out


def observation_mask_from_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    imputed_flags: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Bridge from a pandas frame to a (T, D) observation mask.

    Two sources, because the upstream Data Preparation service may already have filled
    the gaps:

    * ``imputed_flags`` given -- the frame has no NaN left, so a ``notna()`` mask would
      be all ones and the decay mechanism would see a perfectly regular series. The
      mask is ``1 - flag`` instead.
    * ``imputed_flags`` omitted -- fall back to ``notna()``.
    """
    if imputed_flags is not None:
        flags = imputed_flags.reindex(columns=list(columns), fill_value=False)
        return (~flags.to_numpy(dtype=bool)).astype(np.uint8)
    return frame[list(columns)].notna().to_numpy(dtype=bool).astype(np.uint8)


def elapsed_time(index: pd.DatetimeIndex, unit: str = "h") -> np.ndarray:
    """Hours (or another pandas unit) elapsed since the first timestamp."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    deltas = index - index[0]
    return (deltas / pd.Timedelta(1, unit)).to_numpy(dtype=np.float32)
