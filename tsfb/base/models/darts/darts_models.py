# pylint: disable=duplicate-code,too-many-lines
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.base_model import BatchMaker, ModelBase
from tsfb.base.models.darts.darts_persistence import (
    _import_prepare_onnx_inputs,
    load_darts_adapter,
    save_darts_adapter,
)
from tsfb.base.models.darts.darts_utils import DartsUtilsMixin
from tsfb.base.models.darts.grouped_darts_mixin import GroupedDartsMixin, PatchedARIMA
from tsfb.base.schema.darts_config import (
    AllWindowsForecastConfig,
    CovariateSeries,
    CovariateWindowConfig,
    DartsConfig,
    ForecastWindowConfig,
)
from tsfb.base.utils.logging import get_logger
from tsfb.base.utils.utils import train_val_split
from tsfb.conf.darts_conf import DARTS_STAT_MODELS_NO_SERIES_ARG

# Optional runtime for ONNX inference (keep at top-level to satisfy linters)
try:
    import onnxruntime as ORT  # type: ignore
except Exception:  # pragma: no cover
    ORT = None  # type: ignore

logger = get_logger(__name__)


class DartsModelAdapter(GroupedDartsMixin, ModelBase, DartsUtilsMixin):
    """
    Adapter for Darts forecasting models.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model_class: type[ForecastingModel],
        model_args: dict,
        model_name: Optional[str] = None,
        allow_fit_on_eval: bool = False,
        supports_validation: bool = False,
        **kwargs,
    ):
        """
        Initialize the DartsModelAdapter.
        :param model_class: Darts model class.
        :param model_args: Model hyperparameters.
        :param model_name: Name of the model.
        :param allow_fit_on_eval: Allow fitting during evaluation.
        :param supports_validation: Whether validation is supported.
        :param kwargs: Additional arguments.
        """
        super().__init__()
        self.model: Optional[ForecastingModel] = None
        self.model_class: type[ForecastingModel]
        if getattr(model_class, "__name__", "") == "ARIMA":
            self.model_class = PatchedARIMA
        else:
            self.model_class = model_class
        self.config = DartsConfig(**{**model_args, **kwargs})
        self._model_name = model_name
        self.allow_fit_on_eval = allow_fit_on_eval
        self.supports_validation = supports_validation
        self._full_future_cov = None
        self._cov_columns = None
        self.train_ratio_in_tv = 1.0
        self._in_columns: Optional[List[str]] = None
        self._static_transformer: Optional[StaticCovariatesTransformer] = None

        self.onnx_path: Optional[str] = None
        self._onnx_session = None
        self.runtime: str = "darts"  # "darts" | "onnx"
        self.supports_batch_forecast: bool = (
            model_args.get("input_chunk_length") is not None
            and model_args.get("output_chunk_length") is not None
        )

    def _is_deep_model(self) -> bool:
        return isinstance(self.model, TorchForecastingModel)

    def to_onnx(self, path: str, **kwargs) -> None:
        """
        Generic ONNX export entrypoint so onnx_handle can call adapter.to_onnx(...).
        Deep models (TorchForecastingModel) MUST be exportable.
        """
        if self.model is None:
            raise ValueError("Cannot export ONNX: adapter.model is None (not fitted?).")

        if not hasattr(self.model, "to_onnx"):
            raise TypeError(
                f"Model {type(self.model).__name__} does not support to_onnx()."
            )

        # Deep model: always export with params by default
        if self._is_deep_model():
            kwargs.setdefault("export_params", True)

        # Use getattr to call to_onnx so static analyzers don't complain about
        # ForecastingModel not declaring that attribute.
        getattr(self.model, "to_onnx")(path, **kwargs)

        if not os.path.exists(path):
            raise RuntimeError(f"to_onnx() finished but file not found: {path}")

        # update adapter state
        self.onnx_path = path
        self.runtime = "onnx"
        self._onnx_session = None

    # aliases for onnx_handle scanning
    def export_onnx(self, path: str, **kwargs) -> None:
        """Alias to `to_onnx` so external callers can detect export methods."""
        self.to_onnx(path, **kwargs)

    def save_onnx(self, path: str, **kwargs) -> None:
        """Alias to `to_onnx` kept for compatibility with different exporters."""
        self.to_onnx(path, **kwargs)

    @property
    def model_name(self):
        """
        Get the model name.
        :return: Model name string.
        """
        return self._model_name

    def _model_supports_past_covariates(self) -> bool:
        """
        Check if the model supports past covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_past_covariates", False)

    def _model_supports_future_covariates(self) -> bool:
        """
        Check if the model supports future covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_future_covariates", False)

    def _model_supports_static_covariates(self) -> bool:
        """
        Check if the model supports static covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_static_covariates", False)

    def _model_supports_series_input(self) -> bool:
        """
        :return: True if supported series input., else False.
        """
        model_name = type(self.model).__name__
        return model_name not in DARTS_STAT_MODELS_NO_SERIES_ARG

    def _fit_grouped_model(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]],
        *,
        group_cols: List[str],
        time_col: str,
        target_cols: List[str],
        static_cols: Optional[List[str]],
    ) -> "ModelBase":
        grouped_train = train_data.copy()
        grouped_train[time_col] = pd.to_datetime(grouped_train[time_col])
        grouped_train = grouped_train.sort_values(group_cols + [time_col]).reset_index(
            drop=True
        )
        grouped_train = self._merge_static_into_grouped(
            grouped_train, covariates, group_cols, static_cols
        )

        ts_list = self._grouped_frame_to_timeseries(
            grouped_train,
            group_cols=group_cols,
            time_col=time_col,
            value_cols=target_cols,
            static_cols=[c for c in static_cols or [] if c in grouped_train.columns],
        )
        if ts_list is None:
            raise ValueError("Could not build grouped TimeSeries for training.")

        ts_list = self._encode_grouped_static_covariates_if_needed(ts_list)
        self.model = self.model_class(**self.config.get_darts_class_params())

        fit_kwargs: Dict[str, Any] = {"series": ts_list}
        if self._model_supports_past_covariates():
            past_cov = self._grouped_frame_to_timeseries(
                covariates.get("past_covariates") if covariates else None,
                group_cols=group_cols,
                time_col=time_col,
            )
            if past_cov is not None:
                fit_kwargs["past_covariates"] = past_cov
        if self._model_supports_future_covariates():
            future_cov = self._grouped_frame_to_timeseries(
                covariates.get("future_covariates") if covariates else None,
                group_cols=group_cols,
                time_col=time_col,
            )
            if future_cov is not None:
                fit_kwargs["future_covariates"] = future_cov

        with self._suppress_lightning_logs():
            self.model.fit(**fit_kwargs)
        return self

    def _forecast_grouped_model(
        self,
        horizon: int,
        series: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]],
        *,
        group_cols: List[str],
        time_col: str,
        target_cols: List[str],
        static_cols: Optional[List[str]],
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call forecast_fit() before forecast().")

        grouped_series = series.copy()
        grouped_series[time_col] = pd.to_datetime(grouped_series[time_col])
        grouped_series = grouped_series.sort_values(
            group_cols + [time_col]
        ).reset_index(drop=True)
        grouped_series = self._merge_static_into_grouped(
            grouped_series, covariates, group_cols, static_cols
        )

        group_ids = self._group_ids_from_frame(grouped_series, group_cols)
        series_ts = self._grouped_frame_to_timeseries(
            grouped_series,
            group_cols=group_cols,
            time_col=time_col,
            value_cols=target_cols,
            static_cols=[c for c in static_cols or [] if c in grouped_series.columns],
        )
        if series_ts is None:
            raise ValueError("Could not build grouped TimeSeries for forecasting.")
        series_ts = self._transform_grouped_static_covariates_if_needed(series_ts)

        predict_kwargs: Dict[str, Any] = {"n": horizon}
        if self._model_supports_series_input():
            predict_kwargs["series"] = series_ts
        if self._model_supports_past_covariates():
            past_cov = self._grouped_frame_to_timeseries(
                covariates.get("past_covariates") if covariates else None,
                group_cols=group_cols,
                time_col=time_col,
            )
            if past_cov is not None:
                predict_kwargs["past_covariates"] = past_cov
        if self._model_supports_future_covariates():
            future_cov = self._grouped_frame_to_timeseries(
                covariates.get("future_covariates") if covariates else None,
                group_cols=group_cols,
                time_col=time_col,
            )
            if future_cov is not None:
                predict_kwargs["future_covariates"] = future_cov

        with self._suppress_lightning_logs():
            preds_ts = self.model.predict(**predict_kwargs)

        return self._grouped_predictions_to_frame(
            preds_ts,
            group_ids,
            group_cols=group_cols,
            time_col=time_col,
            target_cols=target_cols,
        )

    def forecast_fit(  # noqa: C901  # pylint: disable=too-many-locals
        self,
        train_data: pd.DataFrame,
        *,
        train_ratio_in_tv: float = 1.0,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> "ModelBase":
        """
        Fit the model to training data, automatically handling grouped multi-entity
        frames (with static covariates) or single time series.
        """
        group_cols = self._normalize_group_cols(
            kwargs.get("group_cols") or kwargs.get("group_col") or []
        )
        time_col = kwargs.get("time_col", "date")
        static_cols = kwargs.get("static_cols", None)
        target_cols = kwargs.get("target_cols") or [
            c for c in train_data.columns if c not in group_cols + [time_col]
        ]
        if group_cols and static_cols is not None:
            logger.info("group_cols resolved = %s", group_cols)
            logger.info("static_cols = %s", static_cols)
        if self._is_grouped_frame(train_data, group_cols, time_col):
            logger.info(
                "Detected grouped DataFrame with group_cols=%s, time_col=%s",
                group_cols,
                time_col,
            )
            return self._fit_grouped_model(
                train_data,
                covariates,
                group_cols=group_cols,
                time_col=time_col,
                target_cols=target_cols,
                static_cols=static_cols,
            )
        train_part, valid_part = self._split_and_normalize(
            train_data, train_ratio_in_tv
        )

        # Darts requires main series to be numeric.
        # We strip out datetime/categorical columns
        # (like the original time_col if not dropped).
        # Database engines (e.g. SingleStore DECIMAL)
        # may return numeric data with pandas object dtype.
        # We attempt coercion first before filtering.
        for col in train_part.columns:
            if not pd.api.types.is_numeric_dtype(train_part[col]):
                coerced = pd.to_numeric(train_part[col], errors="coerce")
                if coerced.notna().any():
                    train_part[col] = coerced
                    if valid_part is not None and col in valid_part.columns:
                        valid_part[col] = pd.to_numeric(
                            valid_part[col], errors="coerce"
                        )

        num_cols = train_part.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError(
                f"No numeric columns found in train_data for single series forecasting."
                f"Columns: {train_part.columns.tolist()}, "
                "Dtypes: {train_part.dtypes.to_dict()}"
            )
        train_part = train_part[num_cols]
        if valid_part is not None:
            valid_part = valid_part[[c for c in num_cols if c in valid_part.columns]]

        self._in_columns = train_part.columns.tolist()
        covariates_split = self._prepare_covariates(train_part, valid_part, covariates)
        self.model = self.model_class(**self.config.get_darts_class_params())
        ts_data = self._convert_to_timeseries_grouped(
            train_part, valid_part, covariates_split, covariates
        )
        covs = CovariateSeries(
            past_train=ts_data["pc_ts_train"],
            past_val=ts_data["pc_ts_val"],
            future_train=ts_data["fc_ts_train"],
            future_val=ts_data["fc_ts_val"],
        )
        fit_kwargs = self._build_fit_kwargs(
            ts_data["ts_train"], ts_data["ts_valid"], covs
        )
        with self._suppress_lightning_logs():
            self.model.fit(**fit_kwargs)
        return self

    def _encode_static_covariates_if_needed(
        self, ts_list: List[TimeSeries]
    ) -> List[TimeSeries]:
        """
        Automatically detect categorical static covariates and encode them numerically.
        """
        if not ts_list:
            return ts_list

        static_covs = ts_list[0].static_covariates
        if static_covs is None:
            return ts_list

        if any(
            getattr(static_covs[col], "dtype", None) == "object"
            or str(getattr(static_covs[col], "dtype", "")).startswith("category")
            for col in static_covs.columns
        ):
            logger.info(
                "[AutoEncode] Detected categorical static covariates — applying "
                "StaticCovariatesTransformer."
            )
            self._static_transformer = StaticCovariatesTransformer()
            ts_list = self._static_transformer.fit_transform(ts_list)
        else:
            logger.debug(
                "[AutoEncode] Static covariates already numeric — no encoding applied."
            )
        return ts_list

    def _split_and_normalize(
        self, train_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_part, valid_part = self._split_train_valid(train_data, train_ratio_in_tv)
        return train_part, valid_part

    def _convert_to_timeseries_grouped(
        self,
        train_part: pd.DataFrame,
        valid_part: pd.DataFrame,
        covariates_split: Tuple[
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
        ],
        covariates: Optional[Dict[str, pd.DataFrame]],
    ) -> Dict[str, Optional["TimeSeries"]]:
        covs = CovariateSeries(
            past_train=covariates_split[0],
            past_val=covariates_split[1],
            future_train=covariates_split[2],
            future_val=covariates_split[3],
        )
        ts_group = self._convert_to_timeseries(train_part, valid_part, covs, covariates)

        return {
            "ts_train": ts_group.ts_train,
            "ts_valid": ts_group.ts_valid,
            "pc_ts_train": ts_group.covariates.past_train,
            "pc_ts_val": ts_group.covariates.past_val,
            "fc_ts_train": ts_group.covariates.future_train,
            "fc_ts_val": ts_group.covariates.future_val,
        }

    # Forecast handles both grouped and plain tabular inputs before converting
    # them into Darts TimeSeries objects.
    # pylint: disable=too-many-locals
    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Forecast future values for a given horizon.
        :param horizon: Number of steps to forecast.
        :param series: Input time series as DataFrame.
        :param covariates: Optional covariates dictionary.
        :param kwargs: Additional arguments.
        :return: Forecasted values as numpy array.
        """
        group_cols = self._normalize_group_cols(
            kwargs.get("group_cols") or kwargs.get("group_col") or []
        )
        time_col = kwargs.get("time_col", "date")
        target_cols = kwargs.get("target_cols") or [
            c for c in series.columns if c not in group_cols + [time_col]
        ]
        static_cols = kwargs.get("static_cols")
        if self._is_grouped_frame(series, group_cols, time_col):
            return self._forecast_grouped_model(
                horizon,
                series,
                covariates,
                group_cols=group_cols,
                time_col=time_col,
                target_cols=target_cols,
                static_cols=static_cols,
            )

        _ = kwargs
        if (
            isinstance(series.index, pd.DatetimeIndex)
            and getattr(series.index, "freq", None) is None
        ):
            inferred = pd.infer_freq(series.index)
            if inferred:
                series = series.asfreq(inferred)

        num_cols = series.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError(
                "No numeric columns found in series for single series forecasting!"
            )
        series = series[num_cols]

        past_cov, future_cov = self._extract_forecast_covariates(
            series, horizon, covariates
        )
        sc = covariates.get("static_covariates") if covariates else None
        use_sc = None
        if self._model_supports_static_covariates() and sc is not None:
            if isinstance(sc, pd.DataFrame):
                # Drop identifier columns that are not model features
                id_cols = [
                    c
                    for c in sc.columns
                    if c.endswith("_id") or c in ("group_id", "entity_id")
                ]
                if id_cols:
                    sc = sc.drop(columns=id_cols, errors="ignore")
                if sc.empty:
                    use_sc = None
                else:
                    deduped = sc.drop_duplicates()
                    use_sc = deduped.iloc[[0]].reset_index(drop=True)
            elif isinstance(sc, pd.Series):
                use_sc = sc
            else:
                logger.warning(
                    "[Forecast] Unsupported type for static_covariates: %s",
                    type(sc),
                )
                use_sc = None
        series_ts = TimeSeries.from_dataframe(series, static_covariates=use_sc)
        predict_kwargs = self._build_predict_kwargs(
            horizon, series_ts, past_cov, future_cov
        )
        with self._suppress_lightning_logs():
            assert self.model is not None
            preds_ts = self.model.predict(**predict_kwargs)
        preds = preds_ts.values()

        return preds

    @property
    def supports_probabilistic(self) -> bool:
        """True when the wrapped Darts model was configured with a likelihood function.

        Probabilistic Darts models (e.g. TFTModel with GaussianLikelihood) can return
        multiple samples from predict(n, num_samples=N). Deterministic models always
        return a single sample regardless of num_samples.
        """
        if self.model is None:
            return False
        return (
            isinstance(self.model, TorchForecastingModel)
            and getattr(self.model, "likelihood", None) is not None
        )

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **_kwargs,
    ) -> Optional[np.ndarray]:
        """Draw Monte Carlo samples from a probabilistic Darts model.

        Returns None if the underlying model is deterministic (no likelihood).
        Returns shape (num_samples, horizon) for univariate series.
        """
        if not self.supports_probabilistic:
            return None

        past_cov, future_cov = self._extract_forecast_covariates(
            series, horizon, covariates
        )
        series_ts = TimeSeries.from_dataframe(series)
        predict_kwargs = self._build_predict_kwargs(
            horizon, series_ts, past_cov, future_cov
        )
        predict_kwargs["num_samples"] = num_samples
        with self._suppress_lightning_logs():
            assert self.model is not None
            preds_ts = self.model.predict(**predict_kwargs)

        # preds_ts.all_values() → (horizon, n_components, num_samples)
        all_vals = preds_ts.all_values()  # (T, C, S)
        # Univariate: squeeze C dimension → (T, S) → transpose → (S, T)
        if all_vals.shape[1] == 1:
            return all_vals[:, 0, :].T.astype(np.float32)
        # Multivariate: (S, T, C)
        return all_vals.transpose(2, 0, 1).astype(np.float32)

    def forecast_distribution(
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **_kwargs,
    ) -> Optional[DistributionForecast]:
        """Fit a parametric distribution to Darts probabilistic model samples.

        Returns None for deterministic models.
        """
        samples = self.forecast_probabilistic(
            horizon, series, num_samples=num_samples, covariates=covariates
        )
        if samples is None:
            return None
        return DistributionForecast.from_samples(samples, distribution_name)

    def _split_train_valid(self, train_data, train_ratio):
        """
        Split training data into train and validation sets.
        :param train_data: Input training data.
        :param train_ratio: Ratio for splitting.
        :return: Tuple of (train, validation) DataFrames.
        """
        if self.allow_fit_on_eval or self.model_name == "RegressionModel":
            return train_data, None
        return train_val_split(
            train_data, train_ratio, self.config.get("input_chunk_length", 0)
        )

    def _prepare_covariates(self, train_idx, valid_idx, covariates):
        """
        Prepare covariate data for training and validation.
        :param train_idx: Training data indices.
        :param valid_idx: Validation data indices.
        :param covariates: Covariates dictionary.
        :return: Tuple of (past_cov_train, past_cov_val,
            future_cov_train, future_cov_val).
        """
        pc_train = pc_val = fc_train = fc_val = None
        if covariates:
            pc = covariates.get("past_covariates")
            if pc is not None:
                pc_train = pc.loc[train_idx.index]
                pc_val = pc.loc[valid_idx.index] if valid_idx is not None else None
                self._cov_columns = pc.columns.tolist()
            fc = covariates.get("future_covariates")
            if fc is not None:
                self._full_future_cov = fc
                fc_train = fc
                fc_val = fc
        return pc_train, pc_val, fc_train, fc_val

    def _build_fit_kwargs(
        self,
        ts_train: TimeSeries,
        ts_valid: Optional[TimeSeries],
        covs: CovariateSeries,
    ) -> Dict:
        kwargs = {"series": ts_train}
        if self.supports_validation and ts_valid:
            kwargs["val_series"] = ts_valid
        if self._model_supports_past_covariates():
            if covs.past_train is not None:
                kwargs["past_covariates"] = covs.past_train
            if self.supports_validation and covs.past_val is not None:
                kwargs["val_past_covariates"] = covs.past_val
        if self._model_supports_future_covariates():
            if covs.future_train is not None:
                kwargs["future_covariates"] = covs.future_train
            if self.supports_validation and covs.future_val is not None:
                kwargs["val_future_covariates"] = covs.future_val
        return kwargs

    def _build_predict_kwargs(self, horizon, series_ts, past_cov, future_cov):
        """
        Build keyword arguments for model.predict().
        :param horizon: Forecast horizon.
        :param series_ts: Input TimeSeries.
        :param past_cov: Past covariates TimeSeries.
        :param future_cov: Future covariates TimeSeries.
        :return: Dictionary of predict arguments.
        """
        predict_kwargs = {"n": horizon}
        if self._model_supports_series_input():
            predict_kwargs["series"] = series_ts
        if self._model_supports_past_covariates() and past_cov is not None:
            predict_kwargs["past_covariates"] = past_cov
        if self._model_supports_future_covariates() and future_cov is not None:
            predict_kwargs["future_covariates"] = future_cov
        return predict_kwargs

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Perform batch forecasting over multiple windows.
        """
        eval_maker = self._get_eval_maker(batch_maker)
        n_windows = len(eval_maker.index_list)
        if n_windows == 0:
            return np.empty((0, horizon, getattr(self.model, "n_series", 1)))
        lookback = self.config.get("input_chunk_length")
        output_len = self.config.get("output_chunk_length")
        win_size = lookback + output_len
        x_windows, cov_dict, full_idx = self._get_batch_data(
            batch_maker, eval_maker, n_windows, win_size
        )
        config = AllWindowsForecastConfig(
            eval_maker=eval_maker,
            x_windows=x_windows,
            cov_dict=cov_dict,
            full_idx=full_idx,
            lookback=lookback,
            output_len=output_len,
        )
        return self._forecast_all_windows(config, **kwargs)

    def _get_eval_maker(self, batch_maker: BatchMaker):
        eval_maker = getattr(batch_maker, "_batch_maker", None)
        if eval_maker is None:
            raise ValueError("batch_maker must wrap RollingForecastEvalBatchMaker")
        return eval_maker

    def _forecast_all_windows(
        self, config: AllWindowsForecastConfig, **kwargs
    ) -> np.ndarray:
        preds = []
        win_size = config.lookback + config.output_len
        for i, start in enumerate(config.eval_maker.index_list):
            fw_config = ForecastWindowConfig(
                i=i,
                start=start,
                full_idx=config.full_idx,
                x_window=config.x_windows[i],
                cov_dict=config.cov_dict,
                lookback=config.lookback,
                output_len=config.output_len,
                win_size=win_size,
            )
            preds.append(self._forecast_one_window(fw_config, **kwargs))
        return np.stack([np.asarray(p) for p in preds], axis=0)

    def _forecast_one_window(
        self, config: ForecastWindowConfig, **kwargs
    ) -> np.ndarray:
        window_idx = config.full_idx[config.start - config.win_size : config.start]
        df_in = self._prepare_input_window(config.x_window, window_idx, config.lookback)
        cov_cfg = CovariateWindowConfig(
            config.cov_dict, window_idx, config.i, config.lookback, config.output_len
        )
        cov_input = self._prepare_covariate_window(cov_cfg)

        return self.forecast(
            horizon=config.output_len, series=df_in, covariates=cov_input, **kwargs
        )

    def _prepare_input_window(
        self, x_window: np.ndarray, window_idx: pd.Index, lookback: int
    ) -> pd.DataFrame:
        """
        Prepare the input dataframe for forecasting.
        """
        return pd.DataFrame(
            x_window[-lookback:],
            index=window_idx[-lookback:],
            columns=self._in_columns,
        )

    def _prepare_covariate_window(
        self, config: CovariateWindowConfig
    ) -> Dict[str, pd.DataFrame]:
        cov_input: Dict[str, pd.DataFrame] = {}
        if "past_covariates" in config.cov_dict:
            arr_pc = config.cov_dict["past_covariates"]
            cov_input["past_covariates"] = pd.DataFrame(
                arr_pc[config.window_index, -config.lookback :],
                index=config.window_idx[-config.lookback :],
                columns=self._cov_columns,
            )
        if self._full_future_cov is not None:
            delta = getattr(config.window_idx, "freq", None) or (
                config.window_idx[1] - config.window_idx[0]
            )
            future_idx = pd.date_range(
                start=config.window_idx[-1] + delta,
                periods=config.output_len,
                freq=delta,
            )
            future_cov_df = self._full_future_cov
            if isinstance(future_cov_df.index, pd.DatetimeIndex):
                target_freq = getattr(future_idx, "freq", None) or delta
                if getattr(future_cov_df.index, "freq", None) != target_freq:
                    future_cov_df = future_cov_df.resample(target_freq).ffill()
            cov_input["future_covariates"] = pd.DataFrame(
                future_cov_df.reindex(future_idx),
                index=future_idx,
                columns=future_cov_df.columns,
            )
        return cov_input

    def save_model(self, path: str) -> None:
        """Save the adapter Darts model to the given path."""
        save_darts_adapter(self, path)

    @classmethod
    def load_model(cls, path: str) -> "DartsModelAdapter":
        """Load a DartsModelAdapter instance from the given path."""
        return load_darts_adapter(cls, path)

    def _get_onnx_session(self):
        if self._onnx_session is not None:
            return self._onnx_session

        if not self.onnx_path:
            raise ValueError("No ONNX artifact found (self.onnx_path is None).")

        if ORT is None:  # pragma: no cover - runtime missing
            raise ImportError(
                "onnxruntime is required for ONNX inference but is not installed."
            )

        self._onnx_session = ORT.InferenceSession(self.onnx_path)
        return self._onnx_session

    def _build_onnx_feeds(self, sess, past_feats, future_feats, static_feats):
        """Construct the feeds dict for an ONNX session given feature arrays.

        This isolates logic and keeps _forecast_onnx smaller to satisfy linters.
        """
        input_names = [i.name for i in sess.get_inputs()]
        feeds = {}

        def _maybe_set(name_candidates, arr):
            for n in name_candidates:
                if n in input_names:
                    feeds[n] = arr
                    return True
            return False

        _maybe_set(["x_past", "past_feats", "past"], past_feats)
        _maybe_set(["x_future", "future_feats", "future"], future_feats)
        _maybe_set(["x_static", "static_feats", "static"], static_feats)

        remaining = [n for n in input_names if n not in feeds]
        for n, arr in zip(remaining, (past_feats, future_feats, static_feats)):
            feeds[n] = arr

        return feeds

    def _select_static_cov(self, covariates: Optional[Dict[str, pd.DataFrame]]):
        """Return a single-row DataFrame or Series to use as static covariates.

        Extracts and normalizes the various allowed input shapes into a value
        that can be passed to TimeSeries.from_dataframe.
        """
        sc = covariates.get("static_covariates") if covariates else None
        if sc is None:
            return None
        if isinstance(sc, pd.DataFrame):
            # Drop identifier columns that are not model features
            id_cols = [
                c
                for c in sc.columns
                if c.endswith("_id") or c in ("group_id", "entity_id")
            ]
            if id_cols:
                sc = sc.drop(columns=id_cols, errors="ignore")
            if sc.empty:
                return None
            deduped = sc.drop_duplicates()
            return deduped.iloc[[0]].reset_index(drop=True)
        if isinstance(sc, pd.Series):
            return sc
        return None

    # pylint: disable=too-many-locals
    def _forecast_onnx(  # noqa: C901
        self,
        horizon: int,
        series: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> np.ndarray:
        _ = kwargs
        if self.model is None:
            raise ValueError(
                "adapter.model is None; need fitted torch model to prepare ONNX inputs"
            )

        # enforce chunk limit for ONNX (recommended)
        out_chunk = int(self.config.get("output_chunk_length") or 0)
        if 0 < out_chunk < horizon:
            raise ValueError(
                f"ONNX runtime currently supports horizon <= output_chunk_length "
                f"(got horizon={horizon}, output_chunk_length={out_chunk})."
            )

        # keep same preprocessing logic as your forecast()
        if (
            isinstance(series.index, pd.DatetimeIndex)
            and getattr(series.index, "freq", None) is None
        ):
            inferred = pd.infer_freq(series.index)
            if inferred:
                series = series.asfreq(inferred)

        num_cols = series.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError(
                "No numeric columns found in series for single series forecasting!"
            )
        series = series[num_cols]

        past_cov, future_cov = self._extract_forecast_covariates(
            series, horizon, covariates
        )

        use_sc = (
            self._select_static_cov(covariates)
            if self._model_supports_static_covariates()
            else None
        )

        # prepare ONNX input arrays in one expression to avoid extra locals
        past_feats, future_feats, static_feats = _import_prepare_onnx_inputs()(
            model=self.model,
            series=TimeSeries.from_dataframe(series, static_covariates=use_sc),
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

        sess = self._get_onnx_session()
        feeds = self._build_onnx_feeds(sess, past_feats, future_feats, static_feats)

        yhat = sess.run(None, feeds)[0]  # usually first output is prediction

        # normalize output shape to (horizon, n_targets)
        yhat = np.asarray(yhat)
        # handle batch dimension
        if yhat.ndim == 3 and yhat.shape[0] == 1:
            yhat = yhat[0]
        return yhat
