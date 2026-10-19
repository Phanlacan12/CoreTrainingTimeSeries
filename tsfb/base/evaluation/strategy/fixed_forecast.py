# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict
from tsfb.base.evaluation.strategy.constants import (
    DEFAULT_DISTRIBUTION_TYPE,
    DEFAULT_PROBABILISTIC_NUM_SAMPLES,
    DEFAULT_PROBABILISTIC_OUTPUT_FORMAT,
    DEFAULT_QUANTILE_LEVELS,
    DISTRIBUTION_TYPE_KEY,
    FIXED_FORECAST_REQUIRED_CONFIGS,
    PROBABILISTIC_NUM_SAMPLES_KEY,
    PROBABILISTIC_OUTPUT_FORMAT_KEY,
    QUANTILE_LEVELS_KEY,
    FieldNames,
)
from tsfb.base.evaluation.strategy.fixed_forecast_helpers import (
    CovariateData,
    ForecastConfig,
)
from tsfb.base.evaluation.strategy.forecasting import ForecastingStrategy
from tsfb.base.evaluation.strategy.metric_utils import evaluate_metrics_with_group
from tsfb.base.forecasting.base_forecast import ForecastBase
from tsfb.base.uncertainty import (
    ConformalCalibration,
    apply_symmetric_interval,
    calibrate_from_residual_windows,
    repeat_window_ids,
)
from tsfb.base.utils.data_processing import split_time


class FixedForecast(ForecastingStrategy):
    """Fixed-window forecasting strategy.

    Implements a forecasting strategy with fixed training and test windows.
    Handles data preparation, model training, prediction and evaluation.

    Attributes:
        REQUIRED_CONFIGS (List[str]): Required configuration parameters
    """

    REQUIRED_CONFIGS = list(FIXED_FORECAST_REQUIRED_CONFIGS)

    def __init__(self, strategy_config: Dict[str, Any], evaluator: Any):
        super().__init__(strategy_config, evaluator)

    @staticmethod
    # pylint: disable=too-many-arguments
    def _evaluate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scaler: Any,
        metric_names: List[str],
        groups: Optional[Any] = None,
        interval_lower: Optional[np.ndarray] = None,
        interval_upper: Optional[np.ndarray] = None,
        window_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        return evaluate_metrics_with_group(
            y_true,
            y_pred,
            metric_names,
            scaler=scaler,
            groups=groups,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            window_ids=window_ids,
        )

    # pylint: disable=too-many-locals
    def _get_forecast_config(
        self,
        series: pd.DataFrame,
        series_name: str,
        output_shift: int,
    ) -> ForecastConfig:
        """Get forecast configuration from parameters."""
        horizon = self._get_scalar_config_value("horizon", series_name)
        save_tp = self._get_scalar_config_value("save_true_pred", series_name)
        loader_cfg = self._get_scalar_config_value("data_module_config", series_name)
        uncertainty_cfg = self._get_uncertainty_config()

        # Use simple boolean lookup since default applied in base
        if "return_metrics_per_series" in self.strategy_config:
            return_mps = self._get_scalar_config_value(
                "return_metrics_per_series", series_name
            )
        else:
            return_mps = self.return_metrics_per_series

        _train, _val, _test = self._extract_split_lengths(loader_cfg, series)
        tv_len = _train + _val
        train_ratio = _train / tv_len if tv_len > 0 else 1.0

        use_conformal = uncertainty_cfg is not None
        if use_conformal:
            needed_span = self._prediction_span(horizon, output_shift)
            if _val < needed_span:
                raise ValueError(
                    "Conformal calibration requires validation length >= "
                    "output_chunk_shift + horizon. "
                    f"Got val_len={_val}, output_chunk_shift={output_shift}, "
                    f"horizon={horizon}."
                )
            if _test < needed_span:
                raise ValueError(
                    "Fixed forecast with conformal requires test length >= "
                    "output_chunk_shift + horizon. "
                    f"Got test_len={_test}, output_chunk_shift={output_shift}, "
                    f"horizon={horizon}."
                )
            train_len = _train
            calib_len = _val
            prediction_start = tv_len
            fit_ratio = 1.0
        else:
            total_len = len(series)
            train_len = total_len - self._prediction_span(horizon, output_shift)
            if train_len <= 0:
                raise ValueError("Prediction length exceeds data length")
            calib_len = 0
            prediction_start = train_len
            fit_ratio = train_ratio

        is_multi = series_name == "__default__" or series.shape[1] > 1

        quantile_levels = self.strategy_config.get(
            QUANTILE_LEVELS_KEY, DEFAULT_QUANTILE_LEVELS
        )
        probabilistic_num_samples = self.strategy_config.get(
            PROBABILISTIC_NUM_SAMPLES_KEY, DEFAULT_PROBABILISTIC_NUM_SAMPLES
        )
        probabilistic_output_format = self.strategy_config.get(
            PROBABILISTIC_OUTPUT_FORMAT_KEY, DEFAULT_PROBABILISTIC_OUTPUT_FORMAT
        )
        distribution_type = self.strategy_config.get(
            DISTRIBUTION_TYPE_KEY, DEFAULT_DISTRIBUTION_TYPE
        )

        return ForecastConfig(
            horizon=horizon,
            save_tp=save_tp,
            train_ratio=fit_ratio,
            train_len=train_len,
            calib_len=calib_len,
            prediction_start=prediction_start,
            output_shift=output_shift,
            is_multi=is_multi,
            return_mps=return_mps,
            use_conformal=use_conformal,
            quantile_levels=list(quantile_levels),
            probabilistic_num_samples=int(probabilistic_num_samples),
            probabilistic_output_format=str(probabilistic_output_format),
            distribution_type=str(distribution_type),
        )

    def _normalize_data(self, train_df: pd.DataFrame, loader: Any) -> pd.DataFrame:
        """Apply normalization to training data if scaler exists."""
        if loader is not None:
            if hasattr(loader, "target_scalers") and loader.target_scalers:
                scaler = loader.target_scalers
                return pd.DataFrame(
                    scaler.transform(train_df),
                    columns=train_df.columns,
                    index=train_df.index,
                )
            if hasattr(loader, "predict"):
                result = loader.predict(train_df)
                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, dict) and "data" in result:
                    res = result["data"]
                    if len(res) == len(train_df):
                        res.index = train_df.index
                    return res
                if isinstance(result, pd.DataFrame):
                    if len(result) == len(train_df):
                        result.index = train_df.index
                    return result
        return train_df

    # pylint: disable=too-many-arguments,too-many-locals
    def _fit_conformal_calibrator(
        self,
        approach: ForecastingApproach,
        series: pd.DataFrame,
        config: ForecastConfig,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Any,
    ) -> ConformalCalibration:
        """Collect validation residuals and fit horizon-wise conformal widths."""
        uncertainty_cfg = self._get_uncertainty_config()
        if uncertainty_cfg is None:
            raise ValueError("Conformal calibration requested without uncertainty cfg.")

        target_channel = self._get_scalar_config_value("target_channel", series_name)
        train_df, _ = split_time(series, config.train_len)
        stride = int(uncertainty_cfg.get("calibration_stride") or config.horizon)
        idxs = self._get_index(
            config.train_len,
            config.calib_len,
            config.horizon,
            stride,
            config.output_shift,
        )
        if not idxs:
            raise ValueError("No calibration windows available for conformal fitting.")

        norm_train_df = self._normalize_data(train_df, loader)
        if getattr(loader, "per_entity", False) and getattr(loader, "group_col", None):
            if loader.group_col in norm_train_df.columns:
                norm_train_df = norm_train_df.drop(columns=[loader.group_col])

        cov_train = CovariateData.from_raw_data(
            covariates,
            config.train_len,
            config.horizon,
            config.output_shift,
        ).train
        approach.forecast_fit(
            train_data=norm_train_df,
            covariates=cov_train,
            train_ratio_in_tv=config.train_ratio,
            loader=loader,
        )

        residual_windows: List[np.ndarray] = []
        for start in idxs:
            lookback_df = series.iloc[:start]
            test_df = self._slice_forecast_window(
                series, start, config.horizon, config.output_shift
            )
            lookback = (
                lookback_df if target_channel is None else lookback_df[[target_channel]]
            )
            lookback = self._normalize_data(lookback, loader)
            if getattr(loader, "per_entity", False) and getattr(
                loader, "group_col", None
            ):
                if loader.group_col in lookback.columns:
                    lookback = lookback.drop(columns=[loader.group_col])

            pred_norm = approach.forecast(
                horizon=config.horizon,
                lookback_data=lookback,
                covariates=CovariateData.from_raw_data(
                    covariates,
                    start,
                    config.horizon,
                    config.output_shift,
                ).pred,
            )
            y_true, y_pred, _, _ = self._prepare_prediction(
                pred_norm, test_df, config, series_name, loader
            )
            residual_windows.append(np.abs(y_true - y_pred))

        return calibrate_from_residual_windows(
            np.asarray(residual_windows, dtype=float),
            horizon=config.horizon,
            alpha=float(uncertainty_cfg["alpha"]),
            bonferroni_correction=bool(uncertainty_cfg["bonferroni_correction"]),
            finite_sample_correction=bool(uncertainty_cfg["finite_sample_correction"]),
        )

    # pylint: disable=too-many-arguments
    def _build_interval_payload(
        self,
        *,
        calibration: Optional[ConformalCalibration],
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_lower: Optional[pd.DataFrame],
        y_upper: Optional[pd.DataFrame],
        window_ids: np.ndarray,
        groups: Optional[np.ndarray] = None,
        prob_forecast: Optional[ForecastBase] = None,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "y_true": y_true.to_dict(orient="list"),
            "y_pred": y_pred.to_dict(orient="list"),
            "window_ids": list(map(str, window_ids)),
        }

        lower_df = y_lower
        upper_df = y_upper

        # Derive lower/upper from sample quantiles when conformal is absent
        if prob_forecast is not None and quantile_levels:
            # pylint: disable=import-outside-toplevel
            from tsfb.base.forecasting.distribution_forecast import DistributionForecast

            q_lo = quantile_levels[0]
            q_hi = quantile_levels[-1]
            target_cols = list(y_pred.columns)
            q_dfs = prob_forecast.to_quantile_dfs(
                quantile_levels, index=y_pred.index, target_cols=target_cols
            )
            payload["sample_quantiles"] = {
                k: df.to_dict(orient="list") for k, df in q_dfs.items()
            }
            payload["quantile_levels"] = quantile_levels
            if isinstance(prob_forecast, DistributionForecast):
                payload["distribution"] = prob_forecast.to_payload_dict()
            if lower_df is None:
                lo_key = f"q{int(round(q_lo * 100)):02d}"
                hi_key = f"q{int(round(q_hi * 100)):02d}"
                lower_df = q_dfs.get(lo_key)
                upper_df = q_dfs.get(hi_key)

        if lower_df is not None and upper_df is not None:
            payload["lower"] = lower_df.to_dict(orient="list")
            payload["upper"] = upper_df.to_dict(orient="list")
        if groups is not None:
            payload["groups"] = list(map(str, groups))
        if calibration is not None:
            payload["conformal"] = calibration.to_dict()
        return payload

    def _get_prob_forecast(
        self,
        approach: ForecastingApproach,
        config: ForecastConfig,
        lookback_data: pd.DataFrame,
    ) -> Optional[ForecastBase]:
        """Route to samples or distribution forecast based on config.

        Returns a ForecastBase (ProbabilisticForecast or DistributionForecast),
        or None when the approach does not support probabilistic output.
        """
        fmt = config.probabilistic_output_format

        if fmt == "distribution" and hasattr(approach, "forecast_distribution"):
            return approach.forecast_distribution(  # type: ignore[attr-defined]
                horizon=config.horizon,
                lookback_data=lookback_data,
                distribution_name=config.distribution_type,
                num_samples=config.probabilistic_num_samples,
            )

        if hasattr(approach, "forecast_probabilistic"):
            return approach.forecast_probabilistic(  # type: ignore[attr-defined]
                horizon=config.horizon,
                lookback_data=lookback_data,
                num_samples=config.probabilistic_num_samples,
            )

        return None

    def _fit_and_predict(
        self,
        approach: ForecastingApproach,
        config: ForecastConfig,
        norm_train_df: pd.DataFrame,
        covariate_data: CovariateData,
        loader: Any = None,
    ) -> Tuple[pd.DataFrame, float, float]:
        """Fit model and make predictions."""
        # Fit
        t0 = time.time()
        approach.forecast_fit(
            train_data=norm_train_df,
            covariates=covariate_data.train,
            train_ratio_in_tv=config.train_ratio,
            loader=loader,
        )
        fit_time = time.time() - t0

        # Forecast
        t2 = time.time()
        pred_df = approach.forecast(
            horizon=config.horizon,
            lookback_data=norm_train_df,
            covariates=covariate_data.pred,
        )
        predict_time = time.time() - t2

        return pred_df, fit_time, predict_time

    # Grouped fixed forecast orchestrates split creation, grouped covariate
    # slicing, optional conformal calibration, prediction, and aggregation.
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements
    def _execute_grouped(  # noqa: C901
        self,
        series: pd.DataFrame,
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Any,
    ) -> List[Any]:
        group_col = getattr(loader, "group_col")
        time_col = getattr(loader, "time_col")
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        loader_cfg = self._get_scalar_config_value("data_module_config", series_name)
        uncertainty_cfg = self._get_uncertainty_config()
        use_conformal = uncertainty_cfg is not None
        output_shift = self._resolve_forecast_output_shift(approach)
        prediction_span = self._prediction_span(horizon, output_shift)
        if "return_metrics_per_series" in self.strategy_config:
            return_mps = self._get_scalar_config_value(
                "return_metrics_per_series", series_name
            )
        else:
            return_mps = self.return_metrics_per_series

        grouped = self._grouped_frames(series, group_col, time_col)
        train_bounds: Dict[Any, Tuple[int, int]] = {}
        calib_bounds: Dict[Any, Tuple[int, int]] = {}
        test_bounds: Dict[Any, Tuple[int, int]] = {}
        forecast_origins: Dict[Any, int] = {}
        calibration_windows: List[Tuple[Any, int]] = []
        for gid, g in grouped.items():
            if use_conformal:
                train_len, val_len, test_len = self._extract_split_lengths(
                    loader_cfg, g
                )
                tv_len = train_len + val_len
                if val_len < prediction_span or test_len < prediction_span:
                    self.logger.warning(
                        "Skipping grouped series %r because val_len=%d, "
                        "test_len=%d, output_chunk_shift=%d, horizon=%d",
                        gid,
                        val_len,
                        test_len,
                        output_shift,
                        horizon,
                    )
                    continue
                train_bounds[gid] = (0, train_len)
                calib_bounds[gid] = (train_len, tv_len)
                forecast_origins[gid] = tv_len
                test_bounds[gid] = (
                    tv_len + output_shift,
                    tv_len + output_shift + horizon,
                )
                assert uncertainty_cfg is not None
                stride = int(uncertainty_cfg.get("calibration_stride") or horizon)
                starts = self._get_index(
                    train_len,
                    val_len,
                    horizon,
                    stride,
                    output_shift,
                )
                calibration_windows.extend((gid, start) for start in starts)
            else:
                if len(g) <= prediction_span:
                    self.logger.warning(
                        "Skipping grouped series %r because len=%d <= "
                        "output_chunk_shift + horizon=%d",
                        gid,
                        len(g),
                        prediction_span,
                    )
                    continue
                split_at = len(g) - prediction_span
                train_bounds[gid] = (0, split_at)
                forecast_origins[gid] = split_at
                test_bounds[gid] = (
                    split_at + output_shift,
                    split_at + output_shift + horizon,
                )

        if not train_bounds:
            raise ValueError("No grouped series are long enough for fixed_forecast.")

        full_bounds = {gid: (0, end) for gid, (_, end) in test_bounds.items()}
        train_df = self._slice_grouped_frame(series, train_bounds, group_col, time_col)
        test_df = self._slice_grouped_frame(series, test_bounds, group_col, time_col)
        assert train_df is not None
        assert test_df is not None

        target_cols = (
            [target_channel]
            if target_channel is not None
            else [
                c for c in getattr(loader, "target_cols", []) if c in train_df.columns
            ]
        )
        if not target_cols:
            target_cols = [
                c for c in train_df.columns if c not in {group_col, time_col}
            ]

        norm_train_df = self._transform_grouped_frame(train_df, loader)
        train_ratio = (
            1.0
            if use_conformal
            else self._extract_grouped_train_ratio(
                loader_cfg, series, group_col, time_col
            )
        )
        grouped_kwargs = self._grouped_model_kwargs(loader)

        cov_train: Dict[str, pd.DataFrame] = {}
        cov_pred: Dict[str, pd.DataFrame] = {}
        for name, frame in (covariates or {}).items():
            if name == "static_covariates":
                filtered = self._slice_grouped_frame(
                    frame, train_bounds, group_col, time_col
                )
                if filtered is not None:
                    cov_train[name] = filtered
                    cov_pred[name] = filtered
                continue

            train_source_bounds = (
                full_bounds if name == "future_covariates" else train_bounds
            )
            train_part = self._slice_grouped_frame(
                frame, train_source_bounds, group_col, time_col
            )
            pred_part = self._slice_grouped_frame(
                frame, full_bounds, group_col, time_col
            )
            if train_part is not None:
                cov_train[name] = self._transform_grouped_frame(train_part, loader)
            if pred_part is not None:
                cov_pred[name] = self._transform_grouped_frame(pred_part, loader)

        t0 = time.time()
        approach.forecast_fit(
            train_data=norm_train_df,
            covariates=cov_train,
            train_ratio_in_tv=train_ratio,
            loader=loader,
            **grouped_kwargs,
        )
        calibration = None
        if use_conformal:
            assert uncertainty_cfg is not None
            residual_windows: List[np.ndarray] = []
            grouped_source = self._grouped_frames(series, group_col, time_col)
            for gid, start in calibration_windows:
                g = grouped_source.get(gid)
                if g is None:
                    continue
                lookback_df = g.iloc[:start].copy()
                test_i = self._slice_forecast_window(g, start, horizon, output_shift)
                if len(test_i) < horizon:
                    continue
                pred_norm_i = approach.forecast(
                    horizon=horizon,
                    lookback_data=self._transform_grouped_frame(
                        self._sort_grouped_frame(lookback_df, group_col, time_col),
                        loader,
                    ),
                    covariates=self._filter_grouped_covariates_for_bounds(
                        covariates,
                        gid,
                        start,
                        horizon,
                        output_shift,
                        group_col,
                        time_col,
                        len(g),
                    ),
                    **grouped_kwargs,
                )
                if not isinstance(pred_norm_i, pd.DataFrame):
                    raise TypeError(
                        "Grouped approaches must return a pandas.DataFrame prediction."
                    )
                pred_df_i = self._inverse_transform_grouped_prediction(
                    pred_norm_i,
                    loader,
                    target_cols=target_cols,
                    series_name=series_name,
                )
                y_true_i, y_pred_i, _ = self._align_grouped_truth_pred(
                    test_i, pred_df_i, loader, target_cols
                )
                residual_windows.append(
                    np.abs(
                        y_true_i.to_numpy(dtype=float) - y_pred_i.to_numpy(dtype=float)
                    )
                )
            if not residual_windows:
                raise ValueError(
                    "No calibration windows available for grouped conformal fitting."
                )
            calibration = calibrate_from_residual_windows(
                np.asarray(residual_windows, dtype=float),
                horizon=horizon,
                alpha=float(uncertainty_cfg["alpha"]),
                bonferroni_correction=bool(uncertainty_cfg["bonferroni_correction"]),
                finite_sample_correction=bool(
                    uncertainty_cfg["finite_sample_correction"]
                ),
            )
        fit_time = time.time() - t0

        t1 = time.time()
        pred_norm = approach.forecast(
            horizon=horizon,
            lookback_data=(
                self._transform_grouped_frame(
                    self._slice_grouped_frame(
                        series,
                        {
                            gid: (0, forecast_origin)
                            for gid, forecast_origin in forecast_origins.items()
                        },
                        group_col,
                        time_col,
                    ),
                    loader,
                )
                if use_conformal
                else norm_train_df
            ),
            covariates=cov_pred,
            **grouped_kwargs,
        )
        predict_time = time.time() - t1

        if not isinstance(pred_norm, pd.DataFrame):
            raise TypeError(
                "Grouped approaches must return a pandas.DataFrame prediction."
            )

        pred_df = self._inverse_transform_grouped_prediction(
            pred_norm,
            loader,
            target_cols=target_cols,
            series_name=series_name,
        )
        y_true_df, y_pred_df, groups = self._align_grouped_truth_pred(
            test_df, pred_df, loader, target_cols
        )
        interval_lower_df = None
        interval_upper_df = None
        interval_lower = None
        interval_upper = None
        window_ids = groups.astype(object)
        if calibration is not None:
            interval_lower_df, interval_upper_df = apply_symmetric_interval(
                y_pred_df.copy(),
                calibration.q_hat,
                target_cols=target_cols,
            )
            interval_lower = interval_lower_df[target_cols]
            interval_upper = interval_upper_df[target_cols]

        scores_dict = self._evaluate_metrics(
            y_true=y_true_df,
            y_pred=y_pred_df,
            scaler=None,
            metric_names=self.evaluator.metric_names,
            groups=groups,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            window_ids=window_ids,
        )

        metrics_per_series = {}
        if return_mps:
            metrics_per_series = scores_dict.get("per_series", {})

        interval_payload = self._encode_interval_payload(
            self._build_interval_payload(
                calibration=calibration,
                y_true=y_true_df,
                y_pred=y_pred_df,
                y_lower=interval_lower_df,
                y_upper=interval_upper_df,
                window_ids=window_ids,
                groups=groups,
            ),
            self._get_scalar_config_value("save_true_pred", series_name),
        )

        return scores_dict["global"] + [
            series_name,
            fit_time,
            predict_time,
            interval_payload,
            "",
            json.dumps(metrics_per_series),
        ]

    # pylint: disable=too-many-arguments,too-many-branches,too-many-locals
    def _prepare_prediction(
        self,
        pred_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: ForecastConfig,
        series_name: str,
        loader: Any,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """Prepare prediction and truth data for evaluation."""
        target_cols = getattr(loader, "target_cols", [])
        if not target_cols:
            drop_cols = [
                getattr(loader, "group_col", None),
                getattr(loader, "time_col", None),
            ]
            target_cols = [c for c in test_df.columns if c not in drop_cols]

        if not isinstance(pred_df, pd.DataFrame):
            columns = target_cols if config.is_multi else [series_name]
            if len(pred_df.shape) > 1 and pred_df.shape[1] < len(columns):
                columns = columns[: pred_df.shape[1]]

            pred_df = pd.DataFrame(
                pred_df,
                index=test_df.index,
                columns=columns,
            )

        # Inverse transform prediction if scaler is available
        if loader is not None:
            if hasattr(loader, "inverse_predict"):
                if getattr(loader, "per_entity", False) and hasattr(
                    loader, "group_col"
                ):
                    group_col = loader.group_col
                    if group_col and group_col in test_df.columns:
                        pred_df[group_col] = test_df[group_col].values

                pred_df = loader.inverse_predict(pred_df, series_name=series_name)

                if getattr(loader, "per_entity", False) and hasattr(
                    loader, "group_col"
                ):
                    if loader.group_col in pred_df.columns:
                        pred_df = pred_df.drop(columns=[loader.group_col])
                y_pred = pred_df.to_numpy()
            elif hasattr(loader, "target_scalers") and loader.target_scalers:
                scaler = loader.target_scalers
                y_pred = scaler.inverse_transform(pred_df)
            else:
                y_pred = pred_df.to_numpy()
        else:
            y_pred = pred_df.to_numpy()

        y_true = test_df[: len(y_pred)][target_cols].to_numpy()

        if not config.is_multi:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        return y_true, y_pred, pred_df, target_cols

    # pylint: disable=too-many-locals,too-many-arguments,too-many-statements
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        loader: Any = None,
    ) -> List[Any]:
        """Execute the fixed forecast strategy.

        Main method that orchestrates the forecasting process:
        1. Gets configurations
        2. Prepares data
        3. Trains model and generates predictions
        4. Evaluates results

        Args:
            series: Input time series data
            meta_info: Optional metadata about the series
            approach: Forecasting approach to use
            series_name: Name of the series being forecast
            covariates: Optional covariate data
            loader: Data loader with optional scaler

        Returns:
            List containing metric values and metadata
        """
        if loader is not None and self._is_grouped_mode(series, loader):
            return self._execute_grouped(
                series=series,
                approach=approach,
                series_name=series_name,
                covariates=covariates,
                loader=loader,
            )

        # Get configuration
        output_shift = self._resolve_forecast_output_shift(approach)
        config = self._get_forecast_config(series, series_name, output_shift)
        uncertainty_cfg = self._get_uncertainty_config()
        calibration = None
        prob_forecast: Optional[ForecastBase] = None

        # Split and normalize data
        train_df = series.iloc[: config.train_len]
        if config.use_conformal:
            test_df = self._slice_forecast_window(
                series,
                config.prediction_start,
                config.horizon,
                config.output_shift,
            )
            fit_t0 = time.time()
            calibration = self._fit_conformal_calibrator(
                approach,
                series,
                config,
                series_name,
                covariates,
                loader,
            )
            fit_time = time.time() - fit_t0
            lookback_df = series.iloc[: config.prediction_start]
            lookback = (
                lookback_df
                if self._get_scalar_config_value("target_channel", series_name) is None
                else lookback_df[
                    [self._get_scalar_config_value("target_channel", series_name)]
                ]
            )
            norm_train_df = self._normalize_data(lookback, loader)
            if getattr(loader, "per_entity", False) and getattr(
                loader, "group_col", None
            ):
                if loader.group_col in norm_train_df.columns:
                    norm_train_df = norm_train_df.drop(columns=[loader.group_col])

            covariate_data = CovariateData.from_raw_data(
                covariates,
                config.prediction_start,
                config.horizon,
                config.output_shift,
            )
            t_pred0 = time.time()
            pred_df = approach.forecast(
                horizon=config.horizon,
                lookback_data=norm_train_df,
                covariates=covariate_data.pred,
            )
            predict_time = time.time() - t_pred0
            prob_forecast = self._get_prob_forecast(approach, config, norm_train_df)
        else:
            test_df = self._slice_forecast_window(
                series,
                config.prediction_start,
                config.horizon,
                config.output_shift,
            )
            norm_train_df = self._normalize_data(train_df, loader)

            if getattr(loader, "per_entity", False) and getattr(
                loader, "group_col", None
            ):
                if loader.group_col in norm_train_df.columns:
                    norm_train_df.drop(columns=[loader.group_col], inplace=True)

            covariate_data = CovariateData.from_raw_data(
                covariates,
                config.train_len,
                config.horizon,
                config.output_shift,
            )
            pred_df, fit_time, predict_time = self._fit_and_predict(
                approach, config, norm_train_df, covariate_data, loader=loader
            )
            prob_forecast = self._get_prob_forecast(approach, config, norm_train_df)

        # Prepare data for evaluation
        _, y_pred, pred_df_inv, target_cols = self._prepare_prediction(
            pred_df, test_df, config, series_name, loader
        )

        # Calculate metrics
        groups = None
        group_col_name = None
        if loader and hasattr(loader, "group_col"):
            group_col_name = getattr(loader, "group_col")
            if group_col_name in test_df.columns:
                groups = test_df[: len(y_pred)][group_col_name].to_numpy()

        if groups is not None and not config.is_multi:
            pass

        y_true_for_eval = test_df[: len(y_pred)][target_cols]
        y_pred_for_eval = (
            pred_df_inv[target_cols]
            if all(c in pred_df_inv.columns for c in target_cols)
            else pred_df_inv
        )

        interval_lower_df = None
        interval_upper_df = None
        interval_lower = None
        interval_upper = None
        window_ids = repeat_window_ids(
            [series_name],
            horizon=len(y_true_for_eval),
            n_targets=len(target_cols),
        )
        if calibration is not None and uncertainty_cfg is not None:
            interval_lower_df, interval_upper_df = apply_symmetric_interval(
                y_pred_for_eval.copy(),
                calibration.q_hat,
                target_cols=target_cols,
            )
            interval_lower = interval_lower_df[target_cols]
            interval_upper = interval_upper_df[target_cols]

        scores_dict = self._evaluate_metrics(
            y_true=y_true_for_eval,
            y_pred=y_pred_for_eval,
            scaler=(
                loader.target_scalers
                if loader and hasattr(loader, "target_scalers")
                else None
            ),
            metric_names=self.evaluator.metric_names,
            groups=groups,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            window_ids=window_ids,
        )

        metric_vals = scores_dict["global"]

        metrics_per_series = {}
        if config.return_mps:
            metrics_per_series = scores_dict.get("per_series", {})

        metrics_per_series_json = json.dumps(metrics_per_series)

        # Encode results if needed
        # actual_encoded = self._encode_data(test_df) if config.save_tp else np.nan
        # pred_encoded = self._encode_data(pred_df) if config.save_tp else np.nan

        interval_payload = self._encode_interval_payload(
            self._build_interval_payload(
                calibration=calibration,
                y_true=y_true_for_eval,
                y_pred=y_pred_for_eval,
                y_lower=interval_lower_df,
                y_upper=interval_upper_df,
                window_ids=window_ids,
                groups=groups,
                prob_forecast=prob_forecast,
                quantile_levels=config.quantile_levels,
            ),
            config.save_tp,
        )

        return metric_vals + [
            series_name,
            fit_time,
            predict_time,
            interval_payload,
            "",
            metrics_per_series_json,
        ]

    @staticmethod
    def accepted_metrics() -> List[str]:
        """Get list of accepted metric names.

        Returns:
            List of supported metric names
        """
        return list(get_instantiated_metric_dict().keys())

    @property
    # pylint: disable=duplicate-code
    def field_names(self) -> List[str]:
        """Get names of result fields.

        Returns:
            List of field names for results DataFrame
        """
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.INTERVAL_PAYLOAD,
            # FieldNames.ACTUAL_DATA,
            # FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
            FieldNames.METRICS_PER_SERIES,
        ]
