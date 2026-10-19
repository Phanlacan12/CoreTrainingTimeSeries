from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import GroupedForecastingApproach
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import BatchMaker, ModelBase


class UnivariateToMultivariate(GroupedForecastingApproach):
    """Forecast each series independently.

    Wide mode keeps the original behavior: one model per input column.
    Grouped mode maps one model per ``(group_id, target_col)`` while keeping
    strategy/orchestration batch-based.
    """

    def __init__(self, model_factory: Callable[[], ModelBase]):
        super().__init__()
        self.model_factory = model_factory
        self.models: Dict[Any, ModelBase] = {}

    @staticmethod
    def _strip_grouped_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        drop_keys = {
            "group_col",
            "group_cols",
            "time_col",
            "target_cols",
            "static_cols",
            "pipeline_type",
        }
        return {k: v for k, v in kwargs.items() if k not in drop_keys}

    def _group_frames(
        self,
        frame: pd.DataFrame,
        group_col: str,
        time_col: str,
    ) -> Dict[Any, pd.DataFrame]:
        sorted_frame = self._sort_grouped_frame(frame, group_col, time_col)
        return {
            gid: grp.sort_values(time_col).reset_index(drop=True)
            for gid, grp in sorted_frame.groupby(group_col, sort=True)
        }

    def _select_group_covariates(
        self,
        covariates: Optional[Dict[str, pd.DataFrame]],
        group_id: Any,
    ) -> Dict[str, pd.DataFrame]:
        if not covariates or self.group_col is None or self.time_col is None:
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for name, frame in covariates.items():
            if not isinstance(frame, pd.DataFrame):
                continue

            if name == "static_covariates":
                subset = frame.copy()
                if self.group_col in subset.columns:
                    subset = subset[subset[self.group_col] == group_id]
                    subset = subset.drop(columns=[self.group_col], errors="ignore")
                result[name] = subset.reset_index(drop=True)
                continue

            subset = frame.copy()
            if self.group_col in subset.columns:
                subset = subset[subset[self.group_col] == group_id]
            if self.time_col in subset.columns:
                subset[self.time_col] = pd.to_datetime(subset[self.time_col])
                subset = subset.sort_values(self.time_col)
                subset = subset.set_index(self.time_col, drop=True)
            subset = subset.drop(columns=[self.group_col], errors="ignore")
            result[name] = subset

        return result

    def _group_series_to_single_target(
        self,
        frame: pd.DataFrame,
        target_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        result = frame[[time_col, target_col]].copy()
        result[time_col] = pd.to_datetime(result[time_col])
        result = result.sort_values(time_col).set_index(time_col, drop=True)
        return result[[target_col]]

    def _prediction_index(
        self,
        lookback_index: pd.Index,
        horizon: int,
    ) -> pd.Index:
        if isinstance(lookback_index, pd.DatetimeIndex):
            freq = self._safe_infer_freq(lookback_index)
            return pd.date_range(
                start=lookback_index[-1], periods=horizon + 1, freq=freq
            )[1:]
        step = self._safe_step(lookback_index)
        start = lookback_index[-1] + step
        stop = start + step * horizon
        return pd.RangeIndex(start, stop, step)

    def _ensure_single_target_prediction(
        self,
        pred: Any,
        target_col: str,
        lookback_index: pd.Index,
        horizon: int,
    ) -> pd.DataFrame:
        needs_index_projection = False
        if isinstance(pred, pd.DataFrame):
            df_pred = pred.copy()
            if target_col not in df_pred.columns:
                if len(df_pred.columns) != 1:
                    raise ValueError(
                        (
                            f"Prediction for {target_col!r} must contain "
                            "exactly one column."
                        )
                    )
                df_pred.columns = [target_col]
            needs_index_projection = isinstance(df_pred.index, pd.RangeIndex)
        elif isinstance(pred, pd.Series):
            df_pred = pred.to_frame(name=target_col)
            needs_index_projection = isinstance(df_pred.index, pd.RangeIndex)
        else:
            arr = np.asarray(pred)
            arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
            df_pred = pd.DataFrame(arr, columns=[target_col])
            needs_index_projection = True

        if needs_index_projection or not isinstance(
            df_pred.index, (pd.DatetimeIndex, pd.RangeIndex)
        ):
            df_pred.index = self._prediction_index(lookback_index, horizon)

        return df_pred[[target_col]]

    def _fit_grouped(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]],
        **kwargs: Any,
    ) -> "UnivariateToMultivariate":
        assert self.group_col is not None
        assert self.time_col is not None

        clean_kwargs = self._strip_grouped_kwargs(kwargs)
        self.models = {}
        for group_id, group_df in self._group_frames(
            train_data, self.group_col, self.time_col
        ).items():
            group_covariates = self._select_group_covariates(covariates, group_id)
            for target_col in self.target_cols:
                model = self.model_factory()
                model.forecast_fit(
                    self._group_series_to_single_target(
                        group_df, target_col, self.time_col
                    ),
                    covariates=group_covariates,
                    **clean_kwargs,
                )
                self.models[(group_id, target_col)] = model
        return self

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> "UnivariateToMultivariate":
        if self._configure_grouped_mode(train_data, kwargs):
            return self._fit_grouped(train_data, covariates, **kwargs)

        for col in train_data.columns:
            model = self.model_factory()
            model.forecast_fit(train_data[[col]], covariates=covariates, **kwargs)
            self.models[col] = model
        return self

    # Grouped forecasting stitches per-series models back into one frame.
    # pylint: disable=too-many-locals
    def _forecast_grouped(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Model not fitted")
        assert self.group_col is not None
        assert self.time_col is not None

        clean_kwargs = self._strip_grouped_kwargs(kwargs)
        outputs: List[pd.DataFrame] = []
        for group_id, group_df in self._group_frames(
            lookback_data, self.group_col, self.time_col
        ).items():
            group_covariates = self._select_group_covariates(covariates, group_id)
            pred_cols: Dict[str, np.ndarray] = {}
            pred_index: Optional[pd.Index] = None
            for target_col in self.target_cols:
                model_key = (group_id, target_col)
                if model_key not in self.models:
                    raise RuntimeError(f"Model not fitted for key {model_key!r}")
                model = self.models[model_key]
                lookback_series = self._group_series_to_single_target(
                    group_df, target_col, self.time_col
                )
                pred_df = self._ensure_single_target_prediction(
                    model.forecast(
                        horizon,
                        lookback_series,
                        covariates=group_covariates,
                        **clean_kwargs,
                    ),
                    target_col,
                    lookback_series.index,
                    horizon,
                )
                if pred_index is None:
                    pred_index = pred_df.index
                pred_cols[target_col] = pred_df[target_col].to_numpy()

            if pred_index is None:
                continue
            group_pred = pd.DataFrame(pred_cols, index=pred_index).reset_index()
            group_pred = group_pred.rename(columns={"index": self.time_col})
            group_pred[self.group_col] = group_id
            outputs.append(
                group_pred[[self.time_col, self.group_col] + self.target_cols]
            )

        if not outputs:
            raise RuntimeError("No grouped predictions were produced.")

        return self._finalize_grouped_outputs(outputs)

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        grouped_pred = self._maybe_forecast_grouped(
            horizon,
            lookback_data,
            covariates=covariates,
            **kwargs,
        )
        if grouped_pred is not None:
            return grouped_pred

        if not self.models:
            raise RuntimeError("Model not fitted")

        preds: Dict[str, np.ndarray] = {}
        for col in lookback_data.columns:
            if col not in self.models:
                raise RuntimeError(f"Model not fitted for column {col}")

            model = self.models[col]
            arr = model.forecast(
                horizon, lookback_data[[col]], covariates=covariates, **kwargs
            )
            if hasattr(arr, "values"):
                arr = arr.values
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            preds[col] = arr

        if isinstance(lookback_data.index, pd.DatetimeIndex):
            freq = self._safe_infer_freq(lookback_data.index)
            idx = pd.date_range(
                start=lookback_data.index[-1], periods=horizon + 1, freq=freq
            )[1:]
        else:
            step = self._safe_step(lookback_data.index)
            start = lookback_data.index[-1] + step
            stop = start + step * horizon
            idx = pd.RangeIndex(start, stop, step)
        return pd.DataFrame(preds, index=idx)

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> np.ndarray:
        if self.grouped_mode:
            raise NotImplementedError(
                "Grouped batch_forecast is not supported for UnivariateToMultivariate. "
                "Use grouped sample-based rolling evaluation instead."
            )

        if not self.models:
            raise RuntimeError("Model not fitted")

        eval_orig = getattr(batch_maker, "_batch_maker", None)
        if not isinstance(eval_orig, RollingForecastEvalBatchMaker):
            raise ValueError("batch_maker must wrap RollingForecastEvalBatchMaker")

        try:
            full_index = pd.DatetimeIndex(eval_orig.series.index)
        except Exception:
            full_index = pd.RangeIndex(0, len(eval_orig.series), 1)

        full_np = (
            eval_orig.series
            if isinstance(eval_orig.series, np.ndarray)
            else eval_orig.series.to_numpy()
        )

        all_col_arrs = []
        for col_idx, col_name in enumerate(self.models.keys()):
            col_np = full_np[:, col_idx : col_idx + 1]
            cov_np = self._process_covariates(covariates, col_idx)
            col_df = pd.DataFrame(col_np, full_index, [col_name])
            sub_eval = RollingForecastEvalBatchMaker(
                series=col_df,
                index_list=eval_orig.index_list,
                covariates=cov_np,
            )
            sub_pred = RollingForecastPredictBatchMaker(sub_eval)
            arr = self.models[col_name].batch_forecast(horizon, sub_pred, **kwargs)
            # arr: (n_windows, horizon) or (n_windows, horizon, n_features)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            all_col_arrs.append(arr[:, :, 0:1])  # (n_windows, horizon, 1)

        return np.concatenate(all_col_arrs, axis=2)  # (n_windows, horizon, n_cols)

    def _process_covariates(
        self, covariates: Optional[Dict[str, pd.DataFrame]], col_idx: int
    ) -> Dict[str, pd.DataFrame]:
        """Process covariates for a specific column index into DataFrames."""
        cov_df: Dict[str, pd.DataFrame] = {}
        if covariates:
            for k, df in covariates.items():
                if isinstance(df, pd.DataFrame):
                    cov_df[k] = df.iloc[:, col_idx : col_idx + 1]
                else:
                    cov_df[k] = pd.DataFrame(df[:, col_idx : col_idx + 1])
        return cov_df

    def _create_window_index(
        self, full_index: pd.Index, start: int, horizon: int
    ) -> pd.Index:
        """Create an index for a forecast window."""
        try:
            win_idx = full_index[start : start + horizon]
        except Exception:
            win_idx = pd.RangeIndex(start, start + horizon)
        return win_idx
