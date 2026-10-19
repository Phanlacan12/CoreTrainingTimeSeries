"""Utility functions for metric evaluation in forecasting strategies."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict


# pylint: disable=too-many-arguments
def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: Any,
    metric_names: List[str],
    interval_lower: Optional[np.ndarray] = None,
    interval_upper: Optional[np.ndarray] = None,
    window_ids: Optional[Union[pd.Series, np.ndarray, List]] = None,
) -> List[float]:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        scaler: Scaler used for inverse normalization
        metric_names: List of metric names to compute

    Returns:
        List of computed metric scores
    """
    df = pd.DataFrame({"label": y_true.flatten(), "predicted": y_pred.flatten()})
    if interval_lower is not None:
        df["lower"] = np.asarray(interval_lower).flatten()
    if interval_upper is not None:
        df["upper"] = np.asarray(interval_upper).flatten()
    if window_ids is not None:
        df["window_id"] = np.asarray(window_ids, dtype=object).flatten()

    metric_dict = get_instantiated_metric_dict()
    # Case-insensitive lookup
    metric_dict_lower = {k.lower(): k for k in metric_dict}
    results: List[float] = []

    for name in metric_names:
        resolved = metric_dict_lower.get(name.lower())
        if resolved is None:
            raise ValueError(f"Metric '{name}' is not recognized.")
        metric = metric_dict[resolved]
        # Always pass scaler, metric itself will decide whether to use it
        res = metric.compute_scores(df, "label", "predicted", scaler=scaler)
        results.append(res)

    return results


# pylint: disable=too-many-arguments,too-many-branches
def evaluate_metrics_with_group(
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.DataFrame, np.ndarray],
    metric_names: List[str],
    scaler: Any = None,
    group_col: Optional[str] = None,
    groups: Optional[Union[pd.Series, np.ndarray, List]] = None,
    interval_lower: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    interval_upper: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    window_ids: Optional[Union[pd.Series, np.ndarray, List]] = None,
) -> Dict[str, Any]:
    """Compute evaluation metrics, optionally separated by group.

    Args:
        y_true: Ground truth values. DataFrame or numpy array. should ONLY targets.
        y_pred: Predicted values. DataFrame or numpy array. should ONLY targets.
        metric_names: List of metric names to compute.
        scaler: Scaler used for inverse normalization (optional).
        group_col: Column name to use for grouping if present in y_true (optional).
                   DEPRECATED/Secondary: Prefer passing `groups` argument.
        groups: Array-like of group identifiers corresponding to rows of y_true/y_pred.

    Returns:
        Dictionary containing:
        - 'global': List of global metric scores.
        - 'per_series': Dictionary mapping group/series name to list of metric scores.
    """
    # 1. Prepare flattened arrays for global metrics
    y_true_vals = y_true
    y_pred_vals = y_pred

    if isinstance(y_true, pd.DataFrame):
        if group_col and group_col in y_true.columns:
            y_true_vals = y_true.drop(columns=[group_col])
            if groups is None:
                groups = y_true[group_col]
        else:
            y_true_vals = y_true.values.flatten()

    if isinstance(y_true_vals, pd.DataFrame):
        y_true_vals = y_true_vals.values.flatten()
    if not isinstance(y_true_vals, np.ndarray):
        y_true_vals = np.array(y_true_vals).flatten()
    else:
        y_true_vals = y_true_vals.flatten()

    if isinstance(y_pred, pd.DataFrame):
        y_pred_vals = y_pred.values.flatten()
    else:
        y_pred_vals = np.array(y_pred).flatten()

    lower_vals = None
    upper_vals = None
    if interval_lower is not None:
        lower_vals = (
            interval_lower.values.flatten()
            if isinstance(interval_lower, pd.DataFrame)
            else np.asarray(interval_lower).flatten()
        )
    if interval_upper is not None:
        upper_vals = (
            interval_upper.values.flatten()
            if isinstance(interval_upper, pd.DataFrame)
            else np.asarray(interval_upper).flatten()
        )

    global_scores = evaluate_metrics(
        y_true_vals,
        y_pred_vals,
        scaler,
        metric_names,
        interval_lower=lower_vals,
        interval_upper=upper_vals,
        window_ids=window_ids,
    )
    results: Dict[str, Any] = {"global": global_scores, "per_series": {}}

    # 2. Compute Per-Series (Grouped) Metrics

    # Case A: Explicit Groups provided
    if groups is not None:
        results["per_series"] = _calculate_per_series_explicit_groups(
            y_true,
            y_pred,
            groups,
            metric_names,
            scaler,
            group_col,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            window_ids=window_ids,
        )

    # Case B: Explicit Group Column in y_true (if groups arg not used)
    elif group_col and isinstance(y_true, pd.DataFrame) and group_col in y_true.columns:
        # Pass (already handled via extraction if groups was None)
        pass

    # Case C: Multiple Columns (Wide Format)
    elif isinstance(y_true, pd.DataFrame) and not groups:
        results["per_series"] = _calculate_per_series_wide_format(
            y_true,
            y_pred,
            metric_names,
            scaler,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            window_ids=window_ids,
        )

    return results


# pylint: disable=too-many-arguments,too-many-locals
def _calculate_per_series_explicit_groups(
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.DataFrame, np.ndarray],
    groups: Union[pd.Series, np.ndarray, List],
    metric_names: List[str],
    scaler: Any,
    group_col: Optional[str] = None,
    interval_lower: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    interval_upper: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    window_ids: Optional[Union[pd.Series, np.ndarray, List]] = None,
) -> Dict[str, List[float]]:
    """Helper to calculate per-series metrics when groups are explicitly provided."""
    per_series_results = {}

    y_true_df = pd.DataFrame(y_true).reset_index(drop=True)
    if isinstance(y_true, pd.DataFrame):
        y_true_df = y_true.reset_index(drop=True)
        if group_col and group_col in y_true_df.columns:
            y_true_df = y_true_df.drop(columns=[group_col])

    y_pred_df = (
        y_pred.reset_index(drop=True)
        if isinstance(y_pred, pd.DataFrame)
        else pd.DataFrame(y_pred)
    )
    lower_df = (
        interval_lower.reset_index(drop=True)
        if isinstance(interval_lower, pd.DataFrame)
        else pd.DataFrame(interval_lower)
        if interval_lower is not None
        else None
    )
    upper_df = (
        interval_upper.reset_index(drop=True)
        if isinstance(interval_upper, pd.DataFrame)
        else pd.DataFrame(interval_upper)
        if interval_upper is not None
        else None
    )

    groups_series = pd.Series(groups).reset_index(drop=True)
    window_series = (
        pd.Series(window_ids).reset_index(drop=True) if window_ids is not None else None
    )

    for g_name in groups_series.unique():
        mask = groups_series == g_name
        sub_true = y_true_df[mask].values.flatten()
        sub_pred = y_pred_df[mask].values.flatten()
        sub_lower = lower_df[mask].values.flatten() if lower_df is not None else None
        sub_upper = upper_df[mask].values.flatten() if upper_df is not None else None
        sub_window_ids = (
            window_series[mask].to_numpy().flatten()
            if window_series is not None
            else None
        )
        per_series_results[str(g_name)] = evaluate_metrics(
            sub_true,
            sub_pred,
            scaler,
            metric_names,
            interval_lower=sub_lower,
            interval_upper=sub_upper,
            window_ids=sub_window_ids,
        )

    return per_series_results


# pylint: disable=too-many-locals,too-many-branches
def _calculate_per_series_wide_format(  # noqa: C901
    y_true: pd.DataFrame,
    y_pred: Union[pd.DataFrame, np.ndarray],
    metric_names: List[str],
    scaler: Any,
    interval_lower: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    interval_upper: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    window_ids: Optional[Union[pd.Series, np.ndarray, List]] = None,
) -> Dict[str, List[float]]:
    """Helper to calculate per-series metrics for wide format data."""
    per_series_results = {}
    window_arr = None
    if window_ids is not None:
        window_arr = np.asarray(window_ids, dtype=object).flatten()

    for col in y_true.columns:
        val_true = y_true[col].values
        col_idx = y_true.columns.get_loc(col)

        # Find matching pred
        if isinstance(y_pred, pd.DataFrame) and col in y_pred.columns:
            val_pred = y_pred[col].values
        elif isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] == y_true.shape[1]:
            idx = y_true.columns.get_loc(col)
            val_pred = y_pred.iloc[:, idx].values
        elif not isinstance(y_pred, pd.DataFrame) and y_pred.ndim > 1:
            # numpy
            if y_pred.shape[1] == y_true.shape[1]:
                idx = y_true.columns.get_loc(col)
                val_pred = y_pred[:, idx]
            else:
                val_pred = y_pred  # Broadcast?
        else:
            val_pred = y_pred

        val_lower = None
        if isinstance(interval_lower, pd.DataFrame) and col in interval_lower.columns:
            val_lower = interval_lower[col].values
        elif interval_lower is not None:
            arr_lower = np.asarray(interval_lower)
            if arr_lower.ndim == 1:
                val_lower = arr_lower
            elif arr_lower.ndim > 1 and arr_lower.shape[1] == y_true.shape[1]:
                idx = y_true.columns.get_loc(col)
                val_lower = arr_lower[:, idx]

        val_upper = None
        if isinstance(interval_upper, pd.DataFrame) and col in interval_upper.columns:
            val_upper = interval_upper[col].values
        elif interval_upper is not None:
            arr_upper = np.asarray(interval_upper)
            if arr_upper.ndim == 1:
                val_upper = arr_upper
            elif arr_upper.ndim > 1 and arr_upper.shape[1] == y_true.shape[1]:
                idx = y_true.columns.get_loc(col)
                val_upper = arr_upper[:, idx]

        sub_window_ids = None
        if window_arr is not None:
            if window_arr.size == len(val_true):
                sub_window_ids = window_arr
            elif window_arr.size == len(y_true) * y_true.shape[1]:
                sub_window_ids = window_arr.reshape(len(y_true), y_true.shape[1])[
                    :, col_idx
                ]
            else:
                sub_window_ids = window_arr[: len(val_true)]

        scores = evaluate_metrics(
            val_true,
            val_pred.flatten(),
            scaler,
            metric_names,
            interval_lower=val_lower,
            interval_upper=val_upper,
            window_ids=sub_window_ids,
        )
        per_series_results[str(col)] = scores

    return per_series_results
