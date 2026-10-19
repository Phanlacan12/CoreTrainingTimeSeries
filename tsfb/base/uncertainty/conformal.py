from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalCalibration:
    """Container for symmetric conformal interval widths per forecast horizon."""

    q_hat: np.ndarray
    alpha: float
    alpha_effective: float
    horizon: int
    bonferroni_correction: bool
    finite_sample_correction: bool
    residual_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration metadata into JSON-friendly primitives."""
        return {
            "q_hat": self.q_hat.tolist(),
            "alpha": self.alpha,
            "alpha_effective": self.alpha_effective,
            "horizon": self.horizon,
            "bonferroni_correction": self.bonferroni_correction,
            "finite_sample_correction": self.finite_sample_correction,
            "residual_count": self.residual_count,
        }


def _validate_residual_windows(
    residual_windows: np.ndarray, horizon: int
) -> np.ndarray:
    """Normalize residual windows to shape ``(n_windows, horizon, n_targets)``."""
    arr = np.asarray(residual_windows, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(
            "residual_windows must be a 2D or 3D array shaped as "
            "(n_windows, horizon[, n_targets])."
        )
    if arr.shape[1] != horizon:
        raise ValueError(
            f"Expected residual_windows horizon={horizon}, got shape {arr.shape}."
        )
    return arr


def conformal_quantile(
    residuals: Sequence[float] | np.ndarray,
    *,
    alpha_effective: float,
    finite_sample_correction: bool,
) -> float:
    """Return the conformal quantile for one horizon residual set."""

    values = np.sort(np.asarray(residuals, dtype=float).reshape(-1))
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Cannot calibrate conformal intervals with no residuals.")

    if finite_sample_correction:
        rank = int(np.ceil((values.size + 1) * (1.0 - alpha_effective)))
    else:
        rank = int(np.ceil(values.size * (1.0 - alpha_effective)))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


def calibrate_from_residual_windows(
    residual_windows: np.ndarray,
    *,
    horizon: int,
    alpha: float,
    bonferroni_correction: bool = True,
    finite_sample_correction: bool = True,
) -> ConformalCalibration:
    """Calibrate per-horizon symmetric interval widths from residual windows."""

    arr = _validate_residual_windows(residual_windows, horizon)
    alpha_effective = float(alpha) / horizon if bonferroni_correction else float(alpha)

    q_hat_list: list[float] = [
        conformal_quantile(
            arr[:, h, :].reshape(-1),
            alpha_effective=alpha_effective,
            finite_sample_correction=finite_sample_correction,
        )
        for h in range(horizon)
    ]
    q_hat = np.asarray(q_hat_list, dtype=float)

    return ConformalCalibration(
        q_hat=q_hat,
        alpha=float(alpha),
        alpha_effective=float(alpha_effective),
        horizon=int(horizon),
        bonferroni_correction=bool(bonferroni_correction),
        finite_sample_correction=bool(finite_sample_correction),
        residual_count=int(arr.shape[0]),
    )


def apply_symmetric_interval(
    pred_df: pd.DataFrame,
    q_hat: Sequence[float] | np.ndarray,
    *,
    target_cols: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct lower/upper interval DataFrames from point forecasts."""

    q_arr = np.asarray(q_hat, dtype=float).reshape(-1)
    lower = pred_df.copy()
    upper = pred_df.copy()

    value_cols = list(target_cols) if target_cols is not None else list(pred_df.columns)
    if q_arr.size != len(lower):
        if q_arr.size == 1:
            q_arr = np.repeat(q_arr, len(lower))
        elif len(lower) % q_arr.size == 0:
            q_arr = np.tile(q_arr, len(lower) // q_arr.size)
        else:
            raise ValueError(
                "q_hat length must match prediction length or be scalar. "
                f"Got q_hat={q_arr.size}, prediction_length={len(lower)}."
            )

    for col in value_cols:
        if col not in lower.columns:
            continue
        lower[col] = lower[col].to_numpy(dtype=float) - q_arr
        upper[col] = upper[col].to_numpy(dtype=float) + q_arr

    return lower, upper


def repeat_window_ids(
    window_ids: Sequence[Any],
    *,
    horizon: int,
    n_targets: int = 1,
) -> np.ndarray:
    """Repeat one window id across all horizon/target points in flattened order."""

    ids = np.asarray(list(window_ids), dtype=object)
    if ids.size == 0:
        return np.array([], dtype=object)
    repeat_factor = int(horizon) * int(max(n_targets, 1))
    return np.repeat(ids, repeat_factor)
