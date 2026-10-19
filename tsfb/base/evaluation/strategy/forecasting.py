import abc
import json
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.data_module.dataloader.base_deep_loader import BaseDeepDataLoader
from tsfb.base.data_module.factory import DataPipelineFactory
from tsfb.base.data_module.read_data import SpatioReader
from tsfb.base.evaluation.strategy.constants import (
    FORECASTING_OPTIONAL_CONFIGS,
    FORECASTING_REQUIRED_CONFIGS,
    FORECASTING_TEST_ARTIFACT_ROOT_ENV,
    FieldNames,
)
from tsfb.base.evaluation.strategy.strategy import Strategy
from tsfb.base.mlflow.model_logging import PyFuncLogSpec, clean_name, log_pyfunc_model
from tsfb.base.mlflow.pyfunc_preprocess import PreprocessPyFunc
from tsfb.base.utils.data_processing import split_time
from tsfb.base.utils.logging import get_logger
from tsfb.base.utils.random_utils import fix_all_random_seed, fix_random_seed


class ForecastingStrategy(Strategy, metaclass=abc.ABCMeta):
    """
    The base class for forecasting strategies.
    """

    REQUIRED_CONFIGS = list(FORECASTING_REQUIRED_CONFIGS)
    OPTIONAL_CONFIGS = list(FORECASTING_OPTIONAL_CONFIGS)
    TEST_ARTIFACT_ROOT_ENV = FORECASTING_TEST_ARTIFACT_ROOT_ENV

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._preprocess_logged: bool = False
        self.logger = get_logger("ForecastingStrategy")
        # Handle optional config for returning metrics per series
        strategy_config = args[0] if args else kwargs.get("strategy_config", {})
        self.return_metrics_per_series = strategy_config.get(
            "return_metrics_per_series", False
        )
        if self._get_uncertainty_config() is None and getattr(
            self.evaluator, "metric", None
        ) == ["all"]:
            self.evaluator.drop_interval_metrics()

    @staticmethod
    def _extract_split_lengths(
        loader_cfg: Dict[str, Any], series: pd.DataFrame
    ) -> Tuple[int, int, int]:
        """Extract and calculate train, val, and test lengths from data module config.

        Args:
            loader_cfg: Data module configuration dictionary
            series: The time series DataFrame (to access length and index)

        Returns:
            Tuple of (train_len, val_len, test_len)

        Raises:
            ValueError: If split configurations are invalid
        """
        n = len(series)
        mode = loader_cfg.get("split_mode", "ratio")

        if mode == "ratio":
            train_len, val_len, test_len = ForecastingStrategy._handle_split_mode_ratio(
                loader_cfg, n
            )
        elif mode == "last":
            train_len, val_len, test_len = ForecastingStrategy._handle_split_mode_last(
                loader_cfg, n
            )
        elif mode == "threshold":
            (
                train_len,
                val_len,
                test_len,
            ) = ForecastingStrategy._handle_split_mode_threshold(loader_cfg, series)
        else:
            raise ValueError(f"Invalid split_mode: {mode}")

        if train_len <= 0:
            raise ValueError(
                "Training length must be > 0. Check your split configuration."
            )

        return train_len, val_len, test_len

    @staticmethod
    def _handle_split_mode_ratio(
        loader_cfg: Dict[str, Any], n: int
    ) -> Tuple[int, int, int]:
        split_ratio = loader_cfg.get("split_ratio", {})
        train_frac = split_ratio.get("train")
        val_frac = split_ratio.get("val")
        test_frac = split_ratio.get("test")

        if train_frac is None or val_frac is None or test_frac is None:
            raise ValueError("split_ratio must define 'train', 'val' and 'test'")

        if train_frac + val_frac >= 1.0:
            raise ValueError("Invalid train+val ratio")

        train_len = int(n * train_frac)
        val_len = int(n * val_frac)
        test_len = n - train_len - val_len
        return train_len, val_len, test_len

    @staticmethod
    def _handle_split_mode_last(
        loader_cfg: Dict[str, Any], n: int
    ) -> Tuple[int, int, int]:
        last = loader_cfg.get("split_last", {})
        val_len = int(last.get("val", 0))
        test_len = int(last.get("test", 0))
        if val_len + test_len > n:
            raise ValueError("split_last: val+test exceeds data length")
        train_len = n - val_len - test_len
        return train_len, val_len, test_len

    @staticmethod
    def _get_end_idx(
        t_range: Union[Tuple[Any, Any], List[Any]], df_idx: pd.DatetimeIndex
    ) -> int:
        start = pd.to_datetime(t_range[0])
        end = pd.to_datetime(t_range[1])
        mask = (df_idx >= start) & (df_idx <= end)
        if not mask.any():
            return 0
        return int(np.where(mask)[0][-1]) + 1

    @staticmethod
    # pylint: disable=too-many-locals
    def _handle_split_mode_threshold(
        loader_cfg: Dict[str, Any], series: pd.DataFrame
    ) -> Tuple[int, int, int]:
        thr = loader_cfg.get("split_threshold", {})
        t_train = thr.get("train")
        t_val = thr.get("val")
        t_test = thr.get("test")

        if not isinstance(t_train, (list, tuple)) or len(t_train) != 2:
            raise ValueError("split_threshold requires train=[start_date, end_date]")
        # if not isinstance(t_val, (list, tuple)) or len(t_val) != 2:
        #     raise ValueError("split_threshold requires val=[start_date, end_date]")

        df_idx = series.index
        try:
            if not isinstance(df_idx, pd.DatetimeIndex):
                time_col = loader_cfg.get("time_col") or loader_cfg.get("date_col")
                if time_col and time_col in series.columns:
                    df_idx = pd.DatetimeIndex(pd.to_datetime(series[time_col]))
                else:
                    df_idx = pd.to_datetime(df_idx)
        except Exception as exc:
            raise ValueError(
                "Series index must be convertible to DatetimeIndex "
                "for split_mode='threshold'"
            ) from exc

        train_end = ForecastingStrategy._get_end_idx(t_train, df_idx)  # type: ignore
        if t_val:
            val_end = ForecastingStrategy._get_end_idx(t_val, df_idx)  # type: ignore
        else:
            val_end = train_end

        train_len = train_end
        val_len = max(val_end - train_end, 0)

        n = len(series)
        if t_test is not None:
            if isinstance(t_test, (list, tuple)) and len(t_test) == 2:
                test_end = ForecastingStrategy._get_end_idx(
                    t_test, df_idx
                )  # type: ignore
                test_len = test_end - val_end
            else:
                raise ValueError("split_threshold requires test=[start_date, end_date]")
        else:
            test_len = n - val_end

        test_len = max(test_len, 0)
        return train_len, val_len, test_len

    @staticmethod
    def _get_index(
        train_length: int,
        test_length: int,
        horizon: int,
        stride: int,
        output_shift: int = 0,
    ) -> List[int]:
        """Calculate indices for rolling windows.

        Args:
            train_length: Length of training data
            test_length: Length of test data
            horizon: Forecast horizon
            stride: Step size between windows

        Returns:
            List of starting indices for rolling windows
        """
        test_length = int(test_length) - max(int(output_shift), 0)
        if test_length < int(horizon):
            return []
        data_len = train_length + test_length
        idxs = list(range(train_length, data_len - horizon + 1, stride))
        if (test_length - horizon) % stride != 0:
            idxs.append(data_len - horizon)
        return idxs

    @classmethod
    def _slice_forecast_window(
        cls,
        frame: pd.DataFrame,
        origin: int,
        horizon: int,
        output_shift: int = 0,
    ) -> pd.DataFrame:
        """Slice the evaluation window that matches a shifted forecast output."""
        start = int(origin) + max(int(output_shift), 0)
        end = start + int(horizon)
        return frame.iloc[start:end].copy()

    @staticmethod
    def _prediction_span(horizon: int, output_shift: int = 0) -> int:
        return int(horizon) + max(int(output_shift), 0)

    @staticmethod
    def _read_output_shift(source: Any) -> Optional[int]:
        if source is None:
            return None

        value = (
            source.get("output_chunk_shift")
            if isinstance(source, dict)
            else getattr(source, "output_chunk_shift", None)
        )
        if value is None:
            nested = (
                source.get("model_hyper_params")
                if isinstance(source, dict)
                else getattr(source, "model_hyper_params", None)
            )
            if isinstance(nested, dict):
                value = nested.get("output_chunk_shift")

        if value is None:
            return None
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return None

    def _resolve_forecast_output_shift(self, approach: ForecastingApproach) -> int:
        """Resolve output_chunk_shift from fitted models or model factories."""
        seen: set[int] = set()
        stack: List[Any] = [approach]

        while stack:
            current = stack.pop()
            if current is None:
                continue
            obj_id = id(current)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            for attr_name in ("config", "model_hyper_params"):
                shift = self._read_output_shift(getattr(current, attr_name, None))
                if shift is not None:
                    return shift

            factory = getattr(current, "model_factory", None)
            if factory is not None:
                for candidate in (
                    getattr(factory, "keywords", None) or {},
                    *(getattr(factory, "args", None) or ()),
                ):
                    shift = self._read_output_shift(candidate)
                    if shift is not None:
                        return shift

            stack.append(getattr(current, "model", None))
            nested_models = getattr(current, "models", None)
            if isinstance(nested_models, dict):
                for value in nested_models.values():
                    stack.append(
                        value[0] if isinstance(value, tuple) and value else value
                    )

        return 0

    @staticmethod
    def _sort_grouped_frame(
        frame: pd.DataFrame,
        group_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        if time_col in result.columns:
            result[time_col] = pd.to_datetime(result[time_col])
        sort_cols = [c for c in [group_col, time_col] if c in result.columns]
        if sort_cols:
            result = result.sort_values(sort_cols).reset_index(drop=True)
        return result

    @staticmethod
    def _is_grouped_mode(series: pd.DataFrame, loader: Any) -> bool:
        group_col = getattr(loader, "group_col", None)
        time_col = getattr(loader, "time_col", None)
        return (
            isinstance(series, pd.DataFrame)
            and group_col is not None
            and time_col is not None
            and group_col in series.columns
            and time_col in series.columns
        )

    def _grouped_frames(
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

    def _slice_grouped_frame(
        self,
        frame: Optional[pd.DataFrame],
        bounds: Dict[Any, Tuple[int, int]],
        group_col: str,
        time_col: str,
    ) -> Optional[pd.DataFrame]:
        if frame is None:
            return None
        if group_col not in frame.columns:
            return frame.copy()
        if time_col not in frame.columns:
            return (
                frame[frame[group_col].isin(bounds)]
                .drop_duplicates(subset=[group_col])
                .reset_index(drop=True)
            )

        grouped = self._grouped_frames(frame, group_col, time_col)
        parts = []
        for gid, (start, end) in bounds.items():
            g = grouped.get(gid)
            if g is None:
                continue
            parts.append(g.iloc[start:end].copy())
        if not parts:
            return frame.iloc[0:0].copy()
        return self._sort_grouped_frame(
            pd.concat(parts, ignore_index=True), group_col, time_col
        )

    def _build_grouped_eval_inputs(
        self,
        preprocess: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        raw_df_full = getattr(preprocess, "raw_df_full", None)
        if raw_df_full is None:
            raise ValueError("Grouped preprocess did not expose raw_df_full.")

        group_col = getattr(preprocess, "group_col")
        time_col = getattr(preprocess, "time_col")
        target_cols = [
            c
            for c in getattr(preprocess, "target_cols", [])
            if c in raw_df_full.columns
        ]
        past_cols = [
            c for c in getattr(preprocess, "past_cols", []) if c in raw_df_full.columns
        ]
        future_cols = [
            c
            for c in getattr(preprocess, "future_cols", [])
            if c in raw_df_full.columns
        ]
        static_cols = [
            c
            for c in getattr(preprocess, "static_cols", [])
            if c in raw_df_full.columns
        ]

        ts_data = self._sort_grouped_frame(
            raw_df_full[[time_col, group_col] + target_cols], group_col, time_col
        )

        covariates: Dict[str, pd.DataFrame] = {}
        if past_cols:
            covariates["past_covariates"] = self._sort_grouped_frame(
                raw_df_full[[time_col, group_col] + past_cols], group_col, time_col
            )
        if future_cols:
            covariates["future_covariates"] = self._sort_grouped_frame(
                raw_df_full[[time_col, group_col] + future_cols], group_col, time_col
            )
        if static_cols:
            encoded_static = getattr(preprocess, "get_encoded_static", lambda: None)()
            if encoded_static is not None:
                covariates["static_covariates"] = encoded_static.copy()
            else:
                covariates["static_covariates"] = (
                    raw_df_full[[group_col] + static_cols]
                    .drop_duplicates(subset=[group_col])
                    .reset_index(drop=True)
                )

        return ts_data, covariates

    def _grouped_model_kwargs(self, loader: Any) -> Dict[str, Any]:
        return {
            "group_col": getattr(loader, "group_col", None),
            "group_cols": (
                [getattr(loader, "group_col", None)]
                if getattr(loader, "group_col", None)
                else []
            ),
            "time_col": getattr(loader, "time_col", None),
            "target_cols": list(getattr(loader, "target_cols", []) or []),
            "static_cols": list(getattr(loader, "static_cols", []) or []),
            "pipeline_type": "static_grouped",
        }

    def _transform_grouped_frame(
        self,
        frame: pd.DataFrame,
        loader: Any,
    ) -> pd.DataFrame:
        if hasattr(loader, "transform_frame"):
            return loader.transform_frame(frame)
        return frame

    def _inverse_transform_grouped_prediction(
        self,
        pred_df: pd.DataFrame,
        loader: Any,
        *,
        target_cols: List[str],
        series_name: Optional[str] = None,
    ) -> pd.DataFrame:
        if hasattr(loader, "inverse_transform_frame"):
            return loader.inverse_transform_frame(
                pred_df,
                value_cols=target_cols,
                series_name=series_name,
            )
        return pred_df

    def _extract_grouped_train_ratio(
        self,
        loader_cfg: Dict[str, Any],
        series: pd.DataFrame,
        group_col: str,
        time_col: str,
    ) -> float:
        ratios = []
        for gid, g in self._grouped_frames(series, group_col, time_col).items():
            try:
                train_len, val_len, _ = self._extract_split_lengths(loader_cfg, g)
            except ValueError as exc:
                if "Training length must be > 0" not in str(exc):
                    raise
                self.logger.warning(
                    "Skipping grouped series %r when deriving train_ratio_in_tv "
                    "because the configured train split is empty.",
                    gid,
                )
                continue
            tv_len = train_len + val_len
            if tv_len > 0:
                ratios.append(train_len / tv_len)
        return float(np.mean(ratios)) if ratios else 1.0

    # pylint: disable=too-many-locals
    def _align_grouped_truth_pred(
        self,
        truth_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        loader: Any,
        target_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        group_col = getattr(loader, "group_col")
        time_col = getattr(loader, "time_col")
        truth_sorted = self._sort_grouped_frame(
            truth_df[[time_col, group_col] + target_cols], group_col, time_col
        )
        pred_sorted = self._sort_grouped_frame(
            pred_df[
                [time_col, group_col] + [c for c in target_cols if c in pred_df.columns]
            ],
            group_col,
            time_col,
        )

        truth_renamed = truth_sorted.rename(
            columns={col: f"__truth__{col}" for col in target_cols}
        )
        merged = truth_renamed.merge(
            pred_sorted,
            on=[time_col, group_col],
            how="left",
        )

        missing_cols = [c for c in target_cols if c not in merged.columns]
        if missing_cols:
            raise ValueError(
                "Grouped prediction missing target columns after alignment: "
                f"{missing_cols}"
            )
        if merged[target_cols].isna().any().any():
            missing_rows = merged[merged[target_cols].isna().any(axis=1)][
                [group_col, time_col]
            ].head(5)
            raise ValueError(
                "Grouped prediction missing rows after alignment. "
                f"Sample missing keys: {missing_rows.to_dict(orient='records')}"
            )

        y_true = merged[[f"__truth__{c}" for c in target_cols]].rename(
            columns={f"__truth__{c}": c for c in target_cols}
        )
        y_pred = merged[target_cols]
        groups = merged[group_col].to_numpy()
        return y_true, y_pred, groups

    # pylint: disable=too-many-arguments
    def _filter_grouped_covariates_for_bounds(
        self,
        covariates: Optional[Dict[str, pd.DataFrame]],
        group_value: Any,
        start: int,
        horizon: int,
        output_shift: int,
        group_col: str,
        time_col: str,
        full_length: int,
    ) -> Dict[str, pd.DataFrame]:
        """Slice grouped covariates for one group and one forecast origin."""
        result: Dict[str, pd.DataFrame] = {}
        if not covariates:
            return result

        prediction_end = min(
            start + self._prediction_span(horizon, output_shift), full_length
        )
        full_bounds = {group_value: (0, prediction_end)}
        static_bounds = {group_value: (0, full_length)}

        for name, frame in covariates.items():
            if name == "static_covariates":
                sliced = self._slice_grouped_frame(
                    frame, static_bounds, group_col, time_col
                )
            else:
                sliced = self._slice_grouped_frame(
                    frame, full_bounds, group_col, time_col
                )
            if sliced is not None:
                result[name] = sliced
        return result

    # This is the top-level pipeline entrypoint, so it necessarily coordinates
    # readers, preprocessors, loaders, logging, and grouped/non-grouped flows.
    # pylint: disable=too-many-locals,too-many-statements
    def execute(  # noqa: C901
        self, data_name: str, approach: ForecastingApproach
    ) -> Any:
        """
        Entry point to run forecasting strategy.

        Args:
            data_name: Identifier for the dataset (e.g., file name).
            approach: A ForecastingApproach instance already configured
                with model and horizon.
        """
        deterministic_mode = self._get_scalar_config_value("deterministic", data_name)
        seed = self._get_scalar_config_value("seed", data_name)
        if deterministic_mode == "full":
            fix_all_random_seed(seed)
        elif deterministic_mode == "efficient":
            fix_random_seed(seed)

        pipeline_cfg = self._get_scalar_config_value("data_module_config", data_name)

        reader, preprocess, dataloader = DataPipelineFactory.create(
            pipeline_cfg,
            device=getattr(self, "device", None),
        )

        (
            raw_target,
            raw_past_cov,
            raw_future_cov,
            raw_static_cov,
        ) = reader.load_data()

        if isinstance(reader, SpatioReader) and isinstance(
            dataloader, BaseDeepDataLoader
        ):
            wide_df = raw_target
            assert isinstance(wide_df, pd.DataFrame)

            spatio_splits = preprocess.fit(  # type: ignore[call-arg]
                (wide_df, reader.nodes, reader.od_tensor)
            )

            self._maybe_log_preprocess_model(preprocess, approach)

            train_loader, val_loader, test_loader = dataloader.build_from_splits(
                spatio_splits
            )

            setattr(self, "train_loader", train_loader)
            setattr(self, "val_loader", val_loader)
            setattr(self, "test_loader", test_loader)

            meta_info = pd.Series({"length": len(wide_df)})

            try:
                single_data_results = self._execute(
                    wide_df,
                    meta_info,
                    approach,
                    data_name,
                    covariates={},
                    loader=dataloader,
                )
            except Exception as e:
                log = f"{traceback.format_exc()}\n{e}"
                single_data_results = self.get_default_result(
                    **{FieldNames.LOG_INFO: log}
                )
            return single_data_results

        preprocess.fit(  # type: ignore[call-arg]
            (raw_target, raw_past_cov, raw_future_cov, raw_static_cov)
        )

        self._maybe_log_preprocess_model(preprocess, approach)

        if pipeline_cfg.get("pipeline_type") == "static_grouped":
            ts_data, grouped_covariates = self._build_grouped_eval_inputs(preprocess)
            meta_info = pd.Series(
                {
                    "length": len(ts_data),
                    "group_count": ts_data[getattr(preprocess, "group_col")].nunique(),
                }
            )
            try:
                single_data_results = self._execute(
                    ts_data,
                    meta_info,
                    approach,
                    data_name,
                    covariates=grouped_covariates,
                    loader=preprocess,
                )
            except Exception as e:
                log = f"{traceback.format_exc()}\n{e}"
                single_data_results = self.get_default_result(
                    **{FieldNames.LOG_INFO: log}
                )
            return single_data_results

        ts_data = raw_target

        assert isinstance(ts_data, pd.DataFrame)

        covariates: Dict[str, pd.DataFrame] = {}
        if isinstance(raw_past_cov, pd.DataFrame):
            covariates["past_covariates"] = raw_past_cov
        if isinstance(raw_future_cov, pd.DataFrame):
            covariates["future_covariates"] = raw_future_cov
        if isinstance(raw_static_cov, pd.DataFrame):
            # Use encoded+scaled static covariates if preprocessor supports it;
            # this prevents raw high-magnitude values (e.g. population_density
            # 100-5000) from being mixed with scaled dynamic features (0-1).
            _enc_static = getattr(preprocess, "get_encoded_static", lambda: None)()
            if _enc_static is not None:
                covariates["static_covariates"] = _enc_static
            else:
                covariates["static_covariates"] = raw_static_cov

        meta_info = pd.Series({"length": len(ts_data)})

        try:
            single_data_results = self._execute(
                ts_data,
                meta_info,
                approach,
                data_name,
                covariates=covariates,
                loader=preprocess,
            )
        except Exception as e:
            log = f"{traceback.format_exc()}\n{e}"
            single_data_results = self.get_default_result(**{FieldNames.LOG_INFO: log})
        return single_data_results

    @abc.abstractmethod
    # pylint: disable=too-many-arguments
    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        approach: ForecastingApproach,
        series_name: str,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        loader: Optional[Any] = None,
    ) -> Any:
        """
        Subclass override for strategy logic.

        :param ts_data: The time series data (DataFrame or numpy).
        :param meta_info: Metadata like length.
        :param approach: The ForecastingApproach instance to use for fit & forecast.
        :param data_name: Identifier string.
        """

    def _get_eval_scaler(
        self,
        loader: Any,
        train_valid_data: pd.DataFrame,
        train_ratio_in_tv: float,
    ) -> Optional[StandardScaler]:
        """
        Return a fitted scaler from the loader if available.
        Fallback to StandardScaler().fit() on entire
        target dataframe if loader does not have scalers.
        """
        if hasattr(loader, "inverse_predict"):
            return None

        if hasattr(loader, "target_scalers") and loader.target_scalers:
            return loader.target_scalers

        # Fallback: fit a scaler on the training portion of the target dataframe
        if train_valid_data.empty:
            return None
        train_data, _ = split_time(
            train_valid_data, int(len(train_valid_data) * train_ratio_in_tv)
        )
        numeric_train = train_data.select_dtypes(include=["number"])
        if numeric_train.empty:
            return None
        scaler = StandardScaler().fit(numeric_train)
        return scaler

    def _get_uncertainty_config(self) -> Optional[Dict[str, Any]]:
        """Return normalized uncertainty config or None when disabled."""
        raw_cfg = self.strategy_config.get("uncertainty")
        if not isinstance(raw_cfg, dict):
            return None

        method = str(raw_cfg.get("method", "")).strip().lower()
        if method and method != "conformal":
            return None

        cfg = dict(raw_cfg)
        cfg.setdefault("method", "conformal")
        cfg.setdefault("alpha", 0.1)
        cfg.setdefault("calibration_source", "val")
        cfg.setdefault("bonferroni_correction", True)
        cfg.setdefault("finite_sample_correction", True)
        return cfg

    def _resolve_payload_output_dir(self) -> str:
        """Resolve the directory used to persist interval payload artifacts."""
        configured_dir = self.strategy_config.get("payload_output_dir")
        if configured_dir:
            return os.path.abspath(str(configured_dir))

        artifact_root = os.getenv(self.TEST_ARTIFACT_ROOT_ENV)
        if artifact_root:
            return os.path.abspath(os.path.join(artifact_root, "interval_payloads"))

        return os.path.abspath("interval_payloads")

    def _encode_interval_payload(self, payload: Dict[str, Any], enabled: bool) -> Any:
        """Persist interval debug payload to JSON and return the file path."""
        if not enabled:
            return np.nan
        base_dir = self._resolve_payload_output_dir()
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"interval_payload_{uuid.uuid4().hex}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return file_path

    def _maybe_log_preprocess_model(
        self,
        preprocess: Any,
        approach: ForecastingApproach,
    ) -> None:
        """
        Log the current PreprocessData instance as an MLflow pyfunc model.

        The preprocessor is wrapped in ``PreprocessPyFunc`` and registered as a
        separate MLflow model, using a name derived from the approach name:

            <clean_approach_name>_preprocess

        Logging happens only if:
        - There is an active MLflow run, and
        - This method has not already logged a Preprocess for this strategy instance.
        """
        if self._preprocess_logged:
            return

        if type(approach).__name__ == "OldONNXApproach":
            self.logger.info(
                "Skipping Preprocess pyfunc logging for "
                "OldONNXApproach to avoid re-registration."
            )
            return

        active_run = mlflow.active_run()
        if active_run is None:
            self.logger.info(
                "No active MLflow run detected; skipping Preprocess pyfunc logging."
            )
            return

        experiment = mlflow.get_experiment(active_run.info.experiment_id)
        experiment_name = clean_name(experiment.name)

        approach_name = getattr(approach, "name", "unknown_approach")
        clean_approach = clean_name(approach_name)
        preprocess_model_name = f"{experiment_name}_{clean_approach}_preprocess"

        self.logger.info(
            "Logging Preprocess pyfunc model for approach '%s' as MLflow model '%s' "
            "(run_id=%s)",
            approach_name,
            preprocess_model_name,
            active_run.info.run_id,
        )

        try:
            result = log_pyfunc_model(
                PyFuncLogSpec(
                    model_name=preprocess_model_name,
                    python_model=PreprocessPyFunc(preprocess),
                    register_model=True,
                    registered_model_name=preprocess_model_name,
                    model_role="preprocess",
                    extra_tags={"approach_name": approach_name},
                    extra_metadata={"python_model": "PreprocessPyFunc"},
                )
            )
            self._preprocess_logged = True
            self.logger.info(
                "Logged preprocess model '%s' (artifact_path=%s, uri=%s, run_id=%s, "
                "registry=%s, version=%s).",
                preprocess_model_name,
                result.artifact_path,
                result.model_uri,
                result.run_id,
                result.registered_model_name,
                result.registered_model_version,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to log Preprocess pyfunc model for approach '%s': %s",
                approach_name,
                exc,
            )
