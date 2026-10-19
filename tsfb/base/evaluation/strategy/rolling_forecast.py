# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code,too-many-lines
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict
from tsfb.base.evaluation.strategy.constants import (
    DEFAULT_PROBABILISTIC_NUM_SAMPLES,
    DEFAULT_QUANTILE_LEVELS,
    PROBABILISTIC_NUM_SAMPLES_KEY,
    QUANTILE_LEVELS_KEY,
    ROLLING_FORECAST_REQUIRED_CONFIGS,
    FieldNames,
)
from tsfb.base.evaluation.strategy.forecasting import ForecastingStrategy
from tsfb.base.evaluation.strategy.metric_utils import evaluate_metrics_with_group
from tsfb.base.evaluation.strategy.rolling_batch_maker import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.evaluation.strategy.rolling_forecast_helpers import (
    RollingForecastSupportMixin,
    inverse_transform_prediction,
    transform_dataframe,
)
from tsfb.base.forecasting.probabilistic_forecast import ProbabilisticForecast
from tsfb.base.uncertainty import (
    ConformalCalibration,
    apply_symmetric_interval,
    calibrate_from_residual_windows,
    repeat_window_ids,
)
from tsfb.base.utils.data_processing import split_time


class RollingForecast(RollingForecastSupportMixin, ForecastingStrategy):
    """Rolling forecast evaluation strategy.

    This strategy evaluates forecasting models using a rolling window approach,
    where predictions are made repeatedly by moving forward in time.
    """

    REQUIRED_CONFIGS = list(ROLLING_FORECAST_REQUIRED_CONFIGS)
    OPTIONAL_CONFIGS = ["evaluate_on_normalized"]

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
        loader_cfg = self._get_scalar_config_value("data_module_config", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        uncertainty_cfg = self._get_uncertainty_config()
        use_conformal = uncertainty_cfg is not None
        save_tp = self._get_scalar_config_value("save_true_pred", series_name)
        output_shift = self._resolve_forecast_output_shift(approach)
        prediction_span = self._prediction_span(horizon, output_shift)

        grouped = self._grouped_frames(series, group_col, time_col)
        group_meta: Dict[Any, Dict[str, int]] = {}
        train_bounds: Dict[Any, Tuple[int, int]] = {}
        full_roll_offsets = set()
        calibration_windows: List[Tuple[Any, int]] = []

        for gid, g in grouped.items():
            train_len, val_len, _ = self._extract_split_lengths(loader_cfg, g)
            tv_len = train_len + val_len
            test_len = len(g) - tv_len
            if use_conformal and val_len < prediction_span:
                self.logger.warning(
                    "Skipping grouped series %r because val_len=%d < "
                    "output_chunk_shift + horizon=%d",
                    gid,
                    val_len,
                    prediction_span,
                )
                continue
            if tv_len <= 0 or test_len < prediction_span:
                self.logger.warning(
                    (
                        "Skipping grouped series %r because tv_len=%d, "
                        "test_len=%d, output_chunk_shift=%d, horizon=%d"
                    ),
                    gid,
                    tv_len,
                    test_len,
                    output_shift,
                    horizon,
                )
                continue

            idxs = self._get_index(tv_len, test_len, horizon, stride, output_shift)[
                :num_rollings
            ]
            offsets = [idx - tv_len for idx in idxs]
            if not offsets:
                continue
            group_meta[gid] = {
                "train_len": train_len,
                "val_len": val_len,
                "tv_len": tv_len,
                "test_len": test_len,
                "length": len(g),
            }
            train_bounds[gid] = (0, train_len if use_conformal else tv_len)
            full_roll_offsets.update(offsets)
            if use_conformal:
                assert uncertainty_cfg is not None
                calib_starts = self._get_index(
                    train_len,
                    val_len,
                    horizon,
                    int(uncertainty_cfg.get("calibration_stride") or horizon),
                    output_shift,
                )
                calibration_windows.extend((gid, idx) for idx in calib_starts)

        if not group_meta:
            raise ValueError("No grouped series are eligible for rolling_forecast.")

        roll_offsets = sorted(full_roll_offsets)[:num_rollings]
        train_df = self._slice_grouped_frame(series, train_bounds, group_col, time_col)
        assert train_df is not None
        norm_train_df = self._transform_grouped_frame(train_df, loader)

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

        train_ratio = (
            1.0
            if use_conformal
            else self._extract_grouped_train_ratio(
                loader_cfg, series, group_col, time_col
            )
        )
        grouped_kwargs = self._grouped_model_kwargs(loader)

        cov_train: Dict[str, pd.DataFrame] = {}
        for name, frame in (covariates or {}).items():
            if name == "static_covariates":
                filtered = self._slice_grouped_frame(
                    frame, train_bounds, group_col, time_col
                )
                if filtered is not None:
                    cov_train[name] = filtered
                continue

            source_bounds = (
                train_bounds
                if name == "past_covariates"
                else {gid: (0, meta["length"]) for gid, meta in group_meta.items()}
            )
            train_part = self._slice_grouped_frame(
                frame, source_bounds, group_col, time_col
            )
            if train_part is not None:
                cov_train[name] = self._transform_grouped_frame(train_part, loader)

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
            grouped_source = self._grouped_frames(series, group_col, time_col)
            residual_windows: List[np.ndarray] = []
            for gid, start in calibration_windows:
                g = grouped_source.get(gid)
                if g is None:
                    continue
                test_i = self._slice_forecast_window(g, start, horizon, output_shift)
                if len(test_i) < horizon:
                    continue
                lookback_df = self._sort_grouped_frame(
                    g.iloc[:start].copy(), group_col, time_col
                )
                pred_norm_i = approach.forecast(
                    horizon=horizon,
                    lookback_data=self._transform_grouped_frame(lookback_df, loader),
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
                    (
                        "No grouped validation windows were produced for "
                        "conformal calibration."
                    )
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
        train_time = time.time() - t0

        all_scores: List[Any] = []
        all_per_series: List[Dict] = []
        total_inf_time = 0.0
        payload_records: List[Dict[str, Any]] = []

        for offset in roll_offsets:
            lookback_bounds: Dict[Any, Tuple[int, int]] = {}
            test_bounds: Dict[Any, Tuple[int, int]] = {}
            full_bounds: Dict[Any, Tuple[int, int]] = {}

            for gid, meta in group_meta.items():
                origin = meta["tv_len"] + offset
                test_start = origin + output_shift
                test_end = test_start + horizon
                if test_end > meta["length"]:
                    continue
                lookback_bounds[gid] = (0, origin)
                test_bounds[gid] = (test_start, test_end)
                full_bounds[gid] = (0, test_end)

            if not lookback_bounds:
                continue

            lookback_df = self._slice_grouped_frame(
                series, lookback_bounds, group_col, time_col
            )
            test_df = self._slice_grouped_frame(
                series, test_bounds, group_col, time_col
            )
            assert lookback_df is not None
            assert test_df is not None
            norm_lookback_df = self._transform_grouped_frame(lookback_df, loader)

            cov_pred: Dict[str, pd.DataFrame] = {}
            for name, frame in (covariates or {}).items():
                if name == "static_covariates":
                    filtered = self._slice_grouped_frame(
                        frame, full_bounds, group_col, time_col
                    )
                    if filtered is not None:
                        cov_pred[name] = filtered
                    continue

                source_bounds = (
                    full_bounds
                    if name in {"past_covariates", "future_covariates"}
                    else full_bounds
                )
                pred_part = self._slice_grouped_frame(
                    frame, source_bounds, group_col, time_col
                )
                if pred_part is not None:
                    cov_pred[name] = self._transform_grouped_frame(pred_part, loader)

            t1 = time.time()
            pred_norm = approach.forecast(
                horizon=horizon,
                lookback_data=norm_lookback_df,
                covariates=cov_pred,
                **grouped_kwargs,
            )
            total_inf_time += time.time() - t1

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

            scores_dict = evaluate_metrics_with_group(
                y_true=y_true_df,
                y_pred=y_pred_df,
                metric_names=self.evaluator.metric_names,
                scaler=None,
                groups=groups,
                group_col=None,
                interval_lower=(
                    apply_symmetric_interval(
                        y_pred_df.copy(), calibration.q_hat, target_cols=target_cols
                    )[0][target_cols]
                    if calibration is not None
                    else None
                ),
                interval_upper=(
                    apply_symmetric_interval(
                        y_pred_df.copy(), calibration.q_hat, target_cols=target_cols
                    )[1][target_cols]
                    if calibration is not None
                    else None
                ),
                window_ids=groups.astype(object),
            )
            all_scores.append(scores_dict["global"])
            all_per_series.append(scores_dict["per_series"])
            if save_tp:
                lower_df = upper_df = None
                if calibration is not None:
                    lower_df, upper_df = apply_symmetric_interval(
                        y_pred_df.copy(), calibration.q_hat, target_cols=target_cols
                    )
                payload_records.append(
                    {
                        "offset": int(offset),
                        "groups": groups.astype(str).tolist(),
                        "y_true": y_true_df.to_dict(orient="list"),
                        "y_pred": y_pred_df.to_dict(orient="list"),
                        "lower": (
                            lower_df.to_dict(orient="list")
                            if lower_df is not None
                            else None
                        ),
                        "upper": (
                            upper_df.to_dict(orient="list")
                            if upper_df is not None
                            else None
                        ),
                    }
                )

        if not all_scores:
            raise ValueError("No grouped rolling windows were produced for evaluation.")

        mean_scores = np.mean(np.stack(all_scores), axis=0).tolist()
        final_per_series = {}
        if self.return_metrics_per_series and all_per_series:
            final_per_series = self._aggregate_per_series_scores(all_per_series)

        interval_payload = self._encode_interval_payload(
            self._build_interval_payload(
                calibration=calibration,
                records=payload_records,
            ),
            save_tp,
        )

        return mean_scores + [
            series_name,
            train_time,
            total_inf_time / len(all_scores),
            interval_payload,
            "",
            json.dumps(final_per_series),
        ]

    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        loader: Optional[Any] = None,
    ) -> List[Any]:
        if loader is not None and self._is_grouped_mode(series, loader):
            return self._execute_grouped(
                series=series,
                approach=approach,
                series_name=series_name,
                covariates=covariates,
                loader=loader,
            )
        _model = getattr(approach, "model", None)
        if _model is not None and getattr(_model, "supports_batch_forecast", False):
            return self._eval_batch(
                series, meta_info, approach, series_name, covariates, loader
            )
        return self._eval_sample(
            series, meta_info, approach, series_name, covariates, loader
        )

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Optional[Any],
    ) -> List[Any]:
        # 1. Configuration
        horizon = self._get_scalar_config_value("horizon", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        loader_cfg = self._get_scalar_config_value("data_module_config", series_name)
        evaluate_on_normalized = bool(
            self.strategy_config.get("evaluate_on_normalized", False)
        )
        uncertainty_cfg = self._get_uncertainty_config()
        output_shift = self._resolve_forecast_output_shift(approach)
        prediction_span = self._prediction_span(horizon, output_shift)

        # 2. Data Splits
        _train, _val, _test = self._extract_split_lengths(loader_cfg, series)
        tv_len = _train + _val
        use_conformal = uncertainty_cfg is not None
        train_ratio_in_tv = (
            1.0 if use_conformal else (_train / tv_len if tv_len > 0 else 1.0)
        )

        train_len = _train if use_conformal else tv_len
        test_origin = tv_len
        test_len = len(series) - test_origin
        train_valid, _ = split_time(series, train_len)

        # 3. Preparation: Scaler and Norm
        target_train_valid = (
            train_valid if target_channel is None else train_valid[[target_channel]]
        )

        fallback_scaler = None
        if loader is None or (
            not hasattr(loader, "target_scalers") and not hasattr(loader, "predict")
        ):
            fallback_scaler = self._get_eval_scaler(
                loader, target_train_valid, train_ratio_in_tv
            )

        norm_target = transform_dataframe(
            target_train_valid, loader, fallback_scaler, target_channel
        )
        if getattr(loader, "per_entity", False) and getattr(loader, "group_col", None):
            if getattr(loader, "group_col") in norm_target.columns:
                norm_target = norm_target.drop(columns=[getattr(loader, "group_col")])

        # 4. Training
        t0 = time.time()
        cov_train = self._extract_covariates(covariates, train_len)
        approach.forecast_fit(
            train_data=norm_target,
            covariates=cov_train,
            train_ratio_in_tv=train_ratio_in_tv,
            loader=loader,
        )
        calibration = None
        if use_conformal:
            if _val < prediction_span:
                raise ValueError(
                    "Conformal calibration requires validation length >= "
                    "output_chunk_shift + horizon. "
                    f"Got val_len={_val}, output_chunk_shift={output_shift}, "
                    f"horizon={horizon}."
                )
            calibration = self._calibrate_sample_conformal(
                series,
                approach,
                train_len=train_len,
                val_len=_val,
                horizon=horizon,
                output_shift=output_shift,
                target_channel=target_channel,
                fallback_scaler=fallback_scaler,
                covariates=covariates,
                loader=loader,
                series_name=series_name,
            )
        t1 = time.time()

        # 5. Rolling Execution
        idxs = self._get_index(test_origin, test_len, horizon, stride, output_shift)[
            :num_rollings
        ]

        return self._run_rolling_loop(
            series,
            approach,
            idxs,
            horizon,
            output_shift,
            target_channel,
            fallback_scaler,
            covariates,
            loader,
            series_name,
            t1 - t0,
            calibration=calibration,
            evaluate_on_normalized=evaluate_on_normalized,
        )

    # pylint: disable=too-many-branches,too-many-statements
    def _run_rolling_loop(  # noqa: C901
        self,
        series: pd.DataFrame,
        approach: ForecastingApproach,
        idxs: List[int],
        horizon: int,
        output_shift: int,
        target_channel: Optional[str],
        fallback_scaler: Any,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Optional[Any],
        series_name: str,
        train_time: float,
        calibration: Optional[ConformalCalibration] = None,
        evaluate_on_normalized: bool = False,
    ) -> List[Any]:
        """Execute the rolling forecast loop."""
        all_scores, all_per_series = [], []
        total_inf_time = 0.0
        save_tp = self._get_scalar_config_value("save_true_pred", series_name)
        payload_records: List[Dict[str, Any]] = []

        for start in idxs:
            train_i = series.iloc[:start]
            test_i = self._slice_forecast_window(series, start, horizon, output_shift)

            # Prepare input data
            if target_channel is None:
                target_cols = getattr(loader, "target_cols", [])
                if (
                    getattr(loader, "group_col", None)
                    and getattr(loader, "group_col") not in target_cols
                ):
                    target_cols = list(target_cols) + [getattr(loader, "group_col")]
                if (
                    getattr(loader, "time_col", None)
                    and getattr(loader, "time_col") not in target_cols
                ):
                    target_cols = list(target_cols) + [getattr(loader, "time_col")]

                if not target_cols:
                    target_cols = list(train_i.columns)
                lookback = train_i[target_cols]
            else:
                lookback = train_i[[target_channel]]

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

            # Prediction
            t2 = time.time()
            pred_norm = approach.forecast(
                horizon=horizon,
                lookback_data=lookback,
                covariates=self._extract_covariates_for_prediction(
                    covariates, start, horizon, output_shift
                ),
            )
            total_inf_time += time.time() - t2

            # Probabilistic forecast (optional — foundation models only)
            prob_forecast: Optional[ProbabilisticForecast] = None
            if hasattr(approach, "forecast_probabilistic"):
                _n_samples = int(
                    self.strategy_config.get(
                        PROBABILISTIC_NUM_SAMPLES_KEY, DEFAULT_PROBABILISTIC_NUM_SAMPLES
                    )
                )
                prob_forecast = approach.forecast_probabilistic(  # type: ignore
                    horizon=horizon,
                    lookback_data=lookback,
                    num_samples=_n_samples,
                )

            # Inverse Transform (skip when reporting on normalized scale)
            if evaluate_on_normalized:
                pred_df = pred_norm
            else:
                pred_df = inverse_transform_prediction(
                    pred_norm,
                    test_i,
                    loader,
                    fallback_scaler,
                    target_channel,
                    series_name,
                )

            # Evaluate
            if target_channel is None:
                target_cols = getattr(loader, "target_cols", [])
                if not target_cols:
                    target_cols = list(test_i.columns)

                drop_cols = []
                if getattr(loader, "group_col", None):
                    drop_cols.append(getattr(loader, "group_col"))
                if getattr(loader, "time_col", None):
                    drop_cols.append(getattr(loader, "time_col"))

                final_target_cols = [c for c in target_cols if c not in drop_cols]

                if not final_target_cols:
                    final_target_cols = [
                        c for c in test_i.columns if c not in drop_cols
                    ]

                y_true_df = (
                    transform_dataframe(
                        test_i[final_target_cols],
                        loader,
                        fallback_scaler,
                        target_channel,
                    )
                    if evaluate_on_normalized
                    else test_i[final_target_cols]
                )
            else:
                y_true_df = (
                    transform_dataframe(
                        test_i[[target_channel]],
                        loader,
                        fallback_scaler,
                        target_channel,
                    )
                    if evaluate_on_normalized
                    else test_i[[target_channel]]
                )

            groups = None
            if loader and hasattr(loader, "group_col"):
                gc = getattr(loader, "group_col")
                if gc in test_i.columns:
                    groups = test_i[gc].to_numpy()

            interval_lower_df = None
            interval_upper_df = None
            interval_lower = None
            interval_upper = None
            window_ids = repeat_window_ids(
                [f"{series_name}:{start}"],
                horizon=len(y_true_df),
                n_targets=y_true_df.shape[1],
            )
            if calibration is not None:
                pred_target_df = (
                    pred_df[final_target_cols]
                    if target_channel is None
                    and isinstance(pred_df, pd.DataFrame)
                    and "final_target_cols" in locals()
                    else pred_df
                )
                interval_lower_df, interval_upper_df = apply_symmetric_interval(
                    pred_target_df.copy(),
                    calibration.q_hat,
                    target_cols=(
                        final_target_cols
                        if target_channel is None
                        else [target_channel]
                    ),
                )
                interval_lower = interval_lower_df
                interval_upper = interval_upper_df

            scores_dict = evaluate_metrics_with_group(
                y_true=y_true_df,
                y_pred=(
                    pred_df[final_target_cols]
                    if target_channel is None
                    and isinstance(pred_df, pd.DataFrame)
                    and "final_target_cols" in locals()
                    else pred_df
                ),
                metric_names=self.evaluator.metric_names,
                scaler=None,
                groups=groups,
                group_col=None,
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                window_ids=window_ids,
            )

            all_scores.append(scores_dict["global"])
            all_per_series.append(scores_dict["per_series"])
            if save_tp:
                _q_levels = list(
                    self.strategy_config.get(
                        QUANTILE_LEVELS_KEY, DEFAULT_QUANTILE_LEVELS
                    )
                )
                _lower_df = interval_lower_df
                _upper_df = interval_upper_df
                _sample_quantiles = None
                if prob_forecast is not None:
                    _q_dfs = prob_forecast.to_quantile_dfs(
                        _q_levels,
                        index=y_true_df.index,
                        target_cols=list(y_true_df.columns),
                    )
                    _sample_quantiles = {
                        k: df.to_dict(orient="list") for k, df in _q_dfs.items()
                    }
                    if _lower_df is None:
                        lo_key = f"q{int(round(_q_levels[0] * 100)):02d}"
                        hi_key = f"q{int(round(_q_levels[-1] * 100)):02d}"
                        _lower_df = _q_dfs.get(lo_key)
                        _upper_df = _q_dfs.get(hi_key)

                record: Dict[str, Any] = {
                    "start": int(start),
                    "y_true": y_true_df.to_dict(orient="list"),
                    "y_pred": (
                        pred_df[final_target_cols]
                        if target_channel is None
                        and isinstance(pred_df, pd.DataFrame)
                        and "final_target_cols" in locals()
                        else pred_df
                    ).to_dict(orient="list"),
                    "lower": (
                        _lower_df.to_dict(orient="list")
                        if _lower_df is not None
                        else None
                    ),
                    "upper": (
                        _upper_df.to_dict(orient="list")
                        if _upper_df is not None
                        else None
                    ),
                }
                if _sample_quantiles is not None:
                    record["sample_quantiles"] = _sample_quantiles
                    record["quantile_levels"] = _q_levels
                payload_records.append(record)

        mean_scores = np.mean(np.stack(all_scores), axis=0).tolist()

        final_per_series = {}
        if self.return_metrics_per_series and all_per_series:
            final_per_series = self._aggregate_per_series_scores(all_per_series)

        metrics_per_series_json = json.dumps(final_per_series)
        avg_inf_time = total_inf_time / len(idxs) if idxs else 0.0
        interval_payload = self._encode_interval_payload(
            self._build_interval_payload(
                calibration=calibration,
                records=payload_records,
            ),
            save_tp,
        )

        return mean_scores + [
            series_name,
            train_time,
            avg_inf_time,
            interval_payload,
            "",  # Log info
            metrics_per_series_json,
        ]

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals,too-many-statements
    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]],
        loader: Optional[Any],
    ) -> List[Any]:
        horizon = self._get_scalar_config_value("horizon", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        loader_cfg = self._get_scalar_config_value("data_module_config", series_name)
        evaluate_on_normalized = bool(
            self.strategy_config.get("evaluate_on_normalized", False)
        )
        output_shift = self._resolve_forecast_output_shift(approach)

        if output_shift > 0:
            return self._eval_sample(
                series, meta_info, approach, series_name, covariates, loader
            )

        _train, _val, _test = self._extract_split_lengths(loader_cfg, series)
        tv_len = _train + _val
        train_ratio_in_tv = _train / tv_len if tv_len > 0 else 1.0

        train_len = tv_len
        test_len = len(series) - tv_len
        train_valid, _ = split_time(series, train_len)

        target_train_valid = (
            train_valid if target_channel is None else train_valid[[target_channel]]
        )

        fallback_scaler = None
        if loader is None or (
            not hasattr(loader, "target_scalers") and not hasattr(loader, "predict")
        ):
            fallback_scaler = self._get_eval_scaler(
                loader, target_train_valid, train_ratio_in_tv
            )

        series_target = series if target_channel is None else series[[target_channel]]
        norm_series = transform_dataframe(
            series_target, loader, fallback_scaler, target_channel
        )
        if getattr(loader, "per_entity", False) and getattr(loader, "group_col", None):
            if getattr(loader, "group_col") in norm_series.columns:
                norm_series = norm_series.drop(columns=[getattr(loader, "group_col")])

        cov_train = self._extract_covariates(covariates, train_len)
        t0 = time.time()
        approach.forecast_fit(
            train_data=norm_series.iloc[:train_len],
            covariates=cov_train,
            train_ratio_in_tv=train_ratio_in_tv,
            loader=loader,
        )
        t1 = time.time()

        idxs = self._get_index(train_len, test_len, horizon, stride)[:num_rollings]
        batch_maker = RollingForecastEvalBatchMaker(norm_series, idxs, covariates)
        predict_maker = RollingForecastPredictBatchMaker(batch_maker)

        all_preds: Any = []
        total_inf_time = 0.0
        while predict_maker.has_more_batches():
            t2 = time.time()
            preds_norm = approach.batch_forecast(horizon, predict_maker)
            t3 = time.time()
            total_inf_time += t3 - t2
            all_preds.append(preds_norm)

        all_preds = np.concatenate(all_preds, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]

        if evaluate_on_normalized:
            # targets already come from norm_series — keep preds in normalized space too
            inv_preds = all_preds
        elif fallback_scaler is not None:
            # flatten → (n_windows*horizon, n_series)
            assert isinstance(all_preds, np.ndarray)
            flat = all_preds.reshape(
                -1, all_preds.shape[-1] if all_preds.ndim == 3 else 1
            )
            inv_flat = fallback_scaler.inverse_transform(flat)
            assert isinstance(all_preds, np.ndarray)
            inv_preds = inv_flat.reshape(all_preds.shape)
        elif loader is not None and (
            hasattr(loader, "entity_scalers") or hasattr(loader, "global_scaler")
        ):
            # Evaluate batch is complex to inverse transform dynamically block by block,
            # but targets contains the batches.
            # We inverse transform pred_window individually!
            inv_preds_list = []
            for pred_window, true_window in zip(all_preds, targets):
                # Construct temp DataFrame and use inverse helper
                test_i = pd.DataFrame(
                    true_window,
                    columns=(
                        norm_series.columns
                        if target_channel is None
                        else [target_channel]
                    ),
                )
                # We don't have the original index for the windows,
                # so we just use RangeIndex
                df_res = inverse_transform_prediction(
                    pred_window, test_i, loader, None, target_channel, series_name
                )
                inv_preds_list.append(df_res.to_numpy())
            inv_preds = np.array(inv_preds_list)
        else:
            inv_preds = all_preds

        all_scores = []
        for pred_window, true_window in zip(inv_preds, targets):
            y_true = true_window.flatten()
            y_pred = pred_window.flatten()
            score = self.evaluator.evaluate(
                actual=y_true, predicted=y_pred, scaler=None
            )
            all_scores.append(score)

        mean_scores = np.mean(np.stack(all_scores), axis=0).tolist()
        avg_inf_time = total_inf_time / len(all_scores)

        return mean_scores + [
            series_name,
            t1 - t0,
            avg_inf_time,
            np.nan,
            "",
            json.dumps({}),
        ]

    @staticmethod
    def accepted_metrics() -> List[str]:
        return list(get_instantiated_metric_dict().keys())

    @property
    # pylint: disable=duplicate-code
    def field_names(self) -> List[str]:
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

    def _get_split_lens(
        self, series: pd.DataFrame, meta_info: Optional[pd.Series], tv_ratio: float
    ) -> Tuple[int, int]:
        """Calculate lengths for train and test splits.

        Args:
            series: Input time series data
            meta_info: Optional metadata about the series
            tv_ratio: Train+validation ratio

        Returns:
            Tuple of (train_length, test_length)

        Raises:
            ValueError: If resulting train or test length is <= 0
        """
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError("Length of train/test must be > 0")
        return train_length, test_length
