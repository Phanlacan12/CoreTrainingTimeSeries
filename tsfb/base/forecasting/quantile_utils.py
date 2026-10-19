"""
Utility functions for fitting parametric distributions to quantile values.

Foundation models like Chronos and TimesFM natively output quantile forecasts.
These helpers convert quantile arrays directly to DistributionForecast params
without needing to generate/tile samples first, which is both more efficient
and more accurate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm as _scipy_norm
from scipy.stats import t as _scipy_t


def _z(p: float) -> float:
    """Standard Normal inverse CDF (z-score) at probability p."""
    return float(_scipy_norm.ppf(p))


def fit_normal_from_quantiles(
    q_array: np.ndarray,
    q_levels: list,
) -> dict:
    """
    Fit Normal(loc, scale) parameters from a quantile array.

    Uses all available quantile pairs symmetrically around 0.5 for a
    least-squares estimate of scale, then takes q50 (or median pair) as loc.

    Args:
        q_array: shape (prediction_length, n_quantiles).
        q_levels: list of quantile levels matching q_array columns.

    Returns:
        {"loc": array(T,), "scale": array(T,)}
    """
    levels = np.asarray(q_levels, dtype=float)
    z_scores = np.array([_z(p) for p in levels])  # (n_quantiles,)

    # Least-squares: Q(p) = loc + scale * z(p)
    # → scale = (Q(p) - Q(0.5)) / z(p)  for p ≠ 0.5
    mask = np.abs(levels - 0.5) > 0.01
    if mask.sum() < 2:
        # Fallback: use q90 - q10
        idx_lo = np.argmin(np.abs(levels - 0.1))
        idx_hi = np.argmin(np.abs(levels - 0.9))
        scale = np.clip(
            (q_array[:, idx_hi] - q_array[:, idx_lo]) / (_z(0.9) - _z(0.1)),
            1e-6,
            None,
        )
    else:
        idx_50 = int(np.argmin(np.abs(levels - 0.5)))
        loc_est = q_array[:, idx_50]  # (T,)
        # scale_per_quantile: (T, k) — one estimate per quantile pair
        scale_candidates = (q_array[:, mask] - loc_est[:, None]) / z_scores[mask]
        scale = np.clip(np.mean(scale_candidates, axis=1), 1e-6, None)  # (T,)

    idx_50 = int(np.argmin(np.abs(levels - 0.5)))
    loc = q_array[:, idx_50].copy()
    return {"loc": loc.astype(np.float32), "scale": scale.astype(np.float32)}


def fit_student_t_from_quantiles(  # pylint: disable=too-many-locals
    q_array: np.ndarray,
    q_levels: list,
) -> dict:
    """
    Fit StudentT(df, loc, scale) from a quantile array.

    Estimates df via the inter-quantile ratio method:
      ratio = (Q90 - Q10) / (Q75 - Q25) ≈ t.ppf(0.9)/t.ppf(0.75) for given df
    Then solves for df numerically.

    Falls back to Normal-like parameters when the tails are close to Gaussian.

    Args:
        q_array: shape (prediction_length, n_quantiles).
        q_levels: list of quantile levels matching q_array columns.

    Returns:
        {"df": array(T,), "loc": array(T,), "scale": array(T,)}
    """
    levels = np.asarray(q_levels, dtype=float)
    idx_10 = int(np.argmin(np.abs(levels - 0.1)))
    idx_25 = int(np.argmin(np.abs(levels - 0.25)))
    idx_50 = int(np.argmin(np.abs(levels - 0.5)))
    idx_75 = int(np.argmin(np.abs(levels - 0.75)))
    idx_90 = int(np.argmin(np.abs(levels - 0.9)))

    q10 = q_array[:, idx_10]
    q25 = q_array[:, idx_25]
    q50 = q_array[:, idx_50]
    q75 = q_array[:, idx_75]
    q90 = q_array[:, idx_90]

    loc = q50.copy()
    iqr_outer = np.clip(q90 - q10, 1e-6, None)
    iqr_inner = np.clip(q75 - q25, 1e-6, None)

    # For StudentT, ratio = (t.ppf(0.9) - t.ppf(0.1)) / (t.ppf(0.75) - t.ppf(0.25))
    # is a monotone function of df. Pre-compute reference ratios for df ∈ [2, 200].
    df_grid = np.array([2.1, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 200])
    ref_ratios = np.array(
        [
            (_scipy_t.ppf(0.9, df) - _scipy_t.ppf(0.1, df))
            / (_scipy_t.ppf(0.75, df) - _scipy_t.ppf(0.25, df))
            for df in df_grid
        ]
    )

    observed_ratio = iqr_outer / iqr_inner  # (T,)
    # Interpolate df from observed ratio
    df = np.interp(observed_ratio, ref_ratios[::-1], df_grid[::-1])
    df = np.clip(df, 2.1, 200.0).astype(np.float32)

    # Estimate scale using the outer quantiles
    scale = np.clip(
        iqr_outer / (_scipy_t.ppf(0.9, df) - _scipy_t.ppf(0.1, df)),
        1e-6,
        None,
    ).astype(np.float32)

    return {"df": df, "loc": loc.astype(np.float32), "scale": scale.astype(np.float32)}


def fit_distribution_from_quantiles(
    q_array: np.ndarray,
    q_levels: list,
    distribution_name: str = "normal",
) -> dict:
    """
    Dispatch to the appropriate quantile-based distribution fitter.

    Args:
        q_array: shape (prediction_length, n_quantiles).
        q_levels: quantile levels matching q_array columns.
        distribution_name: 'normal' | 'student_t'.
            NegativeBinomial is not supported via quantile fitting
            — callers should fall back to sample-based fitting.

    Returns:
        dict of distribution parameters.

    Raises:
        ValueError: when distribution_name is not supported.
    """
    if distribution_name == "normal":
        return fit_normal_from_quantiles(q_array, q_levels)
    if distribution_name == "student_t":
        return fit_student_t_from_quantiles(q_array, q_levels)
    raise ValueError(
        f"Quantile-based fitting not supported for '{distribution_name}'. "
        "Use sample-based fitting (from_samples) instead."
    )
