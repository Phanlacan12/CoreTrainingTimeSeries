# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.uncertainty import ConformalCalibration, calibrate_from_residual_windows


class _RollingForecastSupport(Protocol):
    def _get_uncertainty_config(self) -> Optional[Dict[str, Any]]:
        ...

    # pylint: disable=too-many-arguments
    def _get_index(
        self,
        train_length: int,
        test_length: int,
        horizon: int,
        stride: int,
        output_shift: int = 0,
    ) -> List[int]:
        ...

    def _slice_forecast_window(
        self,
        series: pd.DataFrame,
        start: int,
        horizon: int,
        output_shift: int = 0,
    ) -> pd.DataFrame:
        ...

    def _prediction_span(self, horizon: int, output_shift: int = 0) -> int:
        ...

    def _extract_covariates_for_prediction(
        self,
        covariates: Optional[Dict[str, pd.DataFrame]],
        start: int,
        horizon: int,
        output_shift: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        ...


def transform_dataframe(df, loader, fallback_scaler, target_channel=None):
    """Apply the loader or fallback scaler transform while preserving index."""
    if loader is not None:
        if hasattr(loader, "target_scalers") and loader.target_scalers:
            scaler = loader.target_scalers
            return pd.DataFrame(
                scaler.transform(df), index=df.index, columns=df.columns
            )
        if hasattr(loader, "predict"):
            result = loader.predict(df)
            if isinstance(result, tuple):
                result_df = result[0]
            elif isinstance(result, dict):
                result_df = result.get("data", result)
            else:
                result_df = result
            if isinstance(result_df, pd.DataFrame):
                if len(result_df) == len(df):
                    result_df.index = df.index
                if target_channel is not None and target_channel in result_df.columns:
                    return result_df[[target_channel]]
                return result_df

    if fallback_scaler is not None:
        return pd.DataFrame(
            fallback_scaler.transform(df), index=df.index, columns=df.columns
        )
    return df


# pylint: disable=too-many-arguments
def inverse_transform_prediction(
    pred_norm, test_i, loader, fallback_scaler, target_channel, series_name
):
    """Map model outputs back to the original scale and expected columns."""
    target_cols = getattr(loader, "target_cols", [])
    if not target_cols:
        drop_cols = [
            getattr(loader, "group_col", None),
            getattr(loader, "time_col", None),
        ]
        target_cols = [c for c in test_i.columns if c not in drop_cols]

    out_cols = target_cols if target_channel is None else [target_channel]

    if loader is not None:
        if hasattr(loader, "inverse_predict"):
            pred_df = (
                pred_norm.copy()
                if isinstance(pred_norm, pd.DataFrame)
                else pd.DataFrame(pred_norm, index=test_i.index, columns=out_cols)
            )
            if getattr(loader, "per_entity", False) and hasattr(loader, "group_col"):
                group_col = loader.group_col
                if group_col and group_col in test_i.columns:
                    pred_df[group_col] = test_i[group_col].values

            pred_df = loader.inverse_predict(pred_df, series_name=series_name)

            if getattr(loader, "per_entity", False) and hasattr(loader, "group_col"):
                if loader.group_col in pred_df.columns:
                    pred_df = pred_df.drop(columns=[loader.group_col])
            return pred_df
        if hasattr(loader, "target_scalers") and loader.target_scalers:
            scaler = loader.target_scalers
            return pd.DataFrame(
                scaler.inverse_transform(pred_norm),
                index=test_i.index,
                columns=out_cols,
            )

    if fallback_scaler is not None:
        return pd.DataFrame(
            fallback_scaler.inverse_transform(pred_norm),
            index=test_i.index,
            columns=out_cols,
        )

    return (
        pred_norm
        if isinstance(pred_norm, pd.DataFrame)
        else pd.DataFrame(pred_norm, index=test_i.index, columns=out_cols)
    )


class RollingForecastSupportMixin:
    """Reusable helpers shared by rolling forecast execution paths."""

    def _build_interval_payload(
        self: _RollingForecastSupport,
        *,
        calibration: Optional[ConformalCalibration],
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"records": records}
        if calibration is not None:
            payload["conformal"] = calibration.to_dict()
        return payload

    # pylint: disable=too-many-arguments,too-many-locals
    def _calibrate_sample_conformal(
        self: _RollingForecastSupport,
        series: pd.DataFrame,
        approach: ForecastingApproach,
        *,
        train_len: int,
        val_len: int,
        horizon: int,
        output_shift: int,
        target_channel: Optional[str],
        fallback_scaler: Any,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Optional[Any],
        series_name: str,
    ) -> ConformalCalibration:
        uncertainty_cfg = self._get_uncertainty_config()
        if uncertainty_cfg is None:
            raise ValueError("Conformal calibration requested without uncertainty cfg.")

        stride = int(uncertainty_cfg.get("calibration_stride") or horizon)
        idxs = self._get_index(train_len, val_len, horizon, stride, output_shift)
        if not idxs:
            raise ValueError(
                "No validation windows available for conformal calibration."
            )

        residual_windows: List[np.ndarray] = []
        for start in idxs:
            train_i = series.iloc[:start]
            test_i = self._slice_forecast_window(series, start, horizon, output_shift)
            lookback = train_i if target_channel is None else train_i[[target_channel]]
            lookback = transform_dataframe(
                lookback, loader, fallback_scaler, target_channel
            )

            if getattr(loader, "per_entity", False) and getattr(
                loader, "group_col", None
            ):
                if getattr(loader, "group_col") in lookback.columns:
                    lookback = lookback.drop(columns=[getattr(loader, "group_col")])

            if (
                getattr(loader, "time_col", None)
                and getattr(loader, "time_col") in lookback.columns
            ):
                lookback = lookback.drop(columns=[getattr(loader, "time_col")])

            pred_norm = approach.forecast(
                horizon=horizon,
                lookback_data=lookback,
                covariates=self._extract_covariates_for_prediction(
                    covariates, start, horizon, output_shift
                ),
            )
            pred_df = inverse_transform_prediction(
                pred_norm, test_i, loader, fallback_scaler, target_channel, series_name
            )

            if target_channel is None:
                target_cols = getattr(loader, "target_cols", [])
                drop_cols = [
                    getattr(loader, "group_col", None),
                    getattr(loader, "time_col", None),
                ]
                final_target_cols = [c for c in target_cols if c not in drop_cols]
                if not final_target_cols:
                    final_target_cols = [
                        c for c in test_i.columns if c not in drop_cols
                    ]
                y_true_df = test_i[final_target_cols]
                y_pred_df = (
                    pred_df[final_target_cols]
                    if isinstance(pred_df, pd.DataFrame)
                    and all(c in pred_df.columns for c in final_target_cols)
                    else pred_df
                )
            else:
                y_true_df = test_i[[target_channel]]
                y_pred_df = pred_df[[target_channel]]

            residual_windows.append(
                np.abs(
                    y_true_df.to_numpy(dtype=float) - y_pred_df.to_numpy(dtype=float)
                )
            )

        return calibrate_from_residual_windows(
            np.asarray(residual_windows, dtype=float),
            horizon=horizon,
            alpha=float(uncertainty_cfg["alpha"]),
            bonferroni_correction=bool(uncertainty_cfg["bonferroni_correction"]),
            finite_sample_correction=bool(uncertainty_cfg["finite_sample_correction"]),
        )

    def _extract_covariates(
        self: _RollingForecastSupport,
        covariates: Optional[Dict[str, pd.DataFrame]],
        train_len: int,
    ) -> Dict[str, pd.DataFrame]:
        cov_train = {}
        if covariates:
            if "past_covariates" in covariates:
                cov_train["past_covariates"] = covariates["past_covariates"].iloc[
                    :train_len
                ]
            if "future_covariates" in covariates:
                cov_train["future_covariates"] = covariates["future_covariates"]
            if "static_covariates" in covariates:
                cov_train["static_covariates"] = covariates["static_covariates"]
        return cov_train

    def _extract_covariates_for_prediction(
        self: _RollingForecastSupport,
        covariates: Optional[Dict[str, pd.DataFrame]],
        start: int,
        horizon: int,
        output_shift: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        cov_pred = {}
        if covariates:
            prediction_end = start + self._prediction_span(horizon, output_shift)
            if "past_covariates" in covariates:
                cov_pred["past_covariates"] = covariates["past_covariates"].iloc[
                    :prediction_end
                ]
            if "future_covariates" in covariates:
                cov_pred["future_covariates"] = covariates["future_covariates"]
            if "static_covariates" in covariates:
                cov_pred["static_covariates"] = covariates["static_covariates"]
        return cov_pred

    def _aggregate_per_series_scores(
        self: _RollingForecastSupport, all_per_series: List[Dict]
    ) -> Dict:
        final_per_series = {}
        if all_per_series:
            series_keys = set().union(*[d.keys() for d in all_per_series])
            for key in series_keys:
                key_scores = [series[key] for series in all_per_series if key in series]
                if key_scores:
                    final_per_series[key] = np.mean(
                        np.stack(key_scores), axis=0
                    ).tolist()
        return final_per_series
