from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.forecasting.quantile_utils import fit_distribution_from_quantiles
from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.schema.foundation_config import FineTuneConfig

try:
    from chronos import Chronos2Pipeline

    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False


@dataclass
class _InputLayout:
    """Identity and index metadata needed to reconstruct repository outputs."""

    grouped: bool
    item_ids: List[Any]
    target_names: List[str]
    history_indices: List[pd.Index]
    group_col: Optional[str] = None
    time_col: Optional[str] = None


class ChronosAdapter(FoundationModelBase):  # pylint: disable=abstract-method
    """Adapter for Chronos-2 zero-shot, fine-tuned and probabilistic forecasts.

    A wide target frame or a ``static_grouped`` frame is converted to one
    Chronos input per series. ``cross_learning`` is passed through exactly as
    configured; the repository approach still decides whether this adapter is
    called once with all series or once per series.
    """

    supports_batch_forecast: bool = True

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "amazon/chronos-2",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        num_samples: int = 20,
        torch_dtype: str = "bfloat16",
        context_length: Optional[int] = None,
        quantization: Optional[str] = None,
        finetune_config: Optional[dict] = None,
        batch_size: int = 32,
        cross_learning: bool = False,
        use_covariates: bool = False,
        **_kwargs,
    ):
        super().__init__(
            pretrained_model_id=pretrained_model_id,
            device=device,
            inference_mode=inference_mode,
            context_length=context_length,
            torch_dtype=torch_dtype,
            finetune_config=finetune_config,
        )
        self._num_samples = num_samples
        self._quantization = quantization
        self._batch_size = batch_size
        self._cross_learning = bool(cross_learning)
        self._use_covariates = bool(use_covariates)
        self._pipeline: Any = None

    @property
    def model_name(self) -> str:
        return f"chronos[{self._model_id}]"

    @property
    def supports_finetuning(self) -> bool:
        return True

    @property
    def supports_probabilistic(self) -> bool:
        return True

    @property
    def supports_distribution(self) -> bool:
        return True

    @property
    def output_type(self) -> str:
        return "distribution"

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        if not _CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is required for ChronosAdapter.\n"
                'Install with: pip install "chronos-forecasting>=2.1.0"'
            )

        kwargs: dict = {"dtype": self._resolve_torch_dtype()}
        if self._quantization == "int8":
            kwargs["load_in_8bit"] = True
        elif self._quantization == "int4":
            kwargs["load_in_4bit"] = True

        device_map = self._device if self._device != "cpu" else "cpu"
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self._model_id, device_map=device_map, **kwargs
        )

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        **input_meta: Any,
    ) -> Any:
        self._load_pipeline()
        inputs, layout = self._build_inputs(
            series, covariates=covariates, horizon=horizon, **input_meta
        )
        _, means = self._pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            **self._predict_kwargs(inputs, cross_learning=self._cross_learning),
        )
        point = self._stack_means(means, horizon)
        if layout.grouped:
            return self._grouped_point_frame(point, layout, horizon)
        return point[:, 0] if point.shape[1] == 1 else point

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,
        **input_meta: Any,
    ) -> np.ndarray:
        self._load_pipeline()
        inputs, layout = self._build_inputs(
            series, covariates=covariates, horizon=horizon, **input_meta
        )
        q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles, _ = self._pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=q_levels,
            **self._predict_kwargs(inputs, cross_learning=self._cross_learning),
        )
        q_array = self._stack_quantiles(quantiles, horizon)  # (H, N, Q)
        base_samples = np.transpose(q_array, (2, 0, 1))  # (Q, H, N)
        reps = max(1, int(np.ceil(num_samples / len(q_levels))))
        samples = np.tile(base_samples, (reps, 1, 1))[:num_samples]
        if len(layout.item_ids) == 1:
            return samples[:, :, 0].astype(np.float32)
        return samples.astype(np.float32)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
        **input_meta: Any,
    ) -> DistributionForecast:
        q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_array = self._get_quantile_array(
            horizon, series, q_levels, covariates, **input_meta
        )

        if distribution_name == "negative_binomial":
            samples = self.forecast_probabilistic(
                horizon,
                series,
                num_samples=num_samples,
                covariates=covariates,
                **input_meta,
            )
            if samples.ndim == 2:
                return DistributionForecast.from_samples(samples, distribution_name)
            forecasts = [
                DistributionForecast.from_samples(samples[:, :, i], distribution_name)
                for i in range(samples.shape[2])
            ]
            return self._stack_distributions(forecasts)

        if q_array.ndim == 2:
            params = fit_distribution_from_quantiles(
                q_array, q_levels, distribution_name
            )
            return DistributionForecast(distribution_name, params)

        forecasts = []
        for item_idx in range(q_array.shape[1]):
            params = fit_distribution_from_quantiles(
                q_array[:, item_idx, :], q_levels, distribution_name
            )
            forecasts.append(DistributionForecast(distribution_name, params))
        return self._stack_distributions(forecasts)

    def _get_quantile_array(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        q_levels: list,
        covariates: Optional[dict] = None,
        **input_meta: Any,
    ) -> np.ndarray:
        self._load_pipeline()
        inputs, layout = self._build_inputs(
            series, covariates=covariates, horizon=horizon, **input_meta
        )
        quantiles, _ = self._pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=q_levels,
            **self._predict_kwargs(inputs, cross_learning=self._cross_learning),
        )
        result = self._stack_quantiles(quantiles, horizon)
        return result[:, 0, :] if len(layout.item_ids) == 1 else result

    def finetune(
        self,
        train_data: pd.DataFrame,
        config: "FineTuneConfig",
        val_data: Optional[pd.DataFrame] = None,
    ) -> "ChronosAdapter":
        self._load_pipeline()
        context = self._build_context(train_data)
        val_inputs = [self._build_context(val_data)] if val_data is not None else None
        fit_kwargs = {
            "inputs": [context],
            "prediction_length": 1,
            "finetune_mode": config.mode,
            "learning_rate": config.learning_rate,
            "context_length": config.context_length or self._context_length,
            "num_steps": config.epochs * 10,
            "batch_size": config.batch_size,
            "disable_data_parallel": True,
            **config.extra_kwargs,
        }
        if val_inputs is not None:
            fit_kwargs["validation_inputs"] = val_inputs
        if config.checkpoint_dir:
            fit_kwargs["output_dir"] = config.checkpoint_dir
        self._pipeline = self._pipeline.fit(**fit_kwargs)
        return self

    def _infer_contexts(self, contexts: list, horizon: int) -> np.ndarray:
        """Batch rolling origins for compute only; never cross-learn origins."""
        self._load_pipeline()
        _, means = self._pipeline.predict_quantiles(
            inputs=contexts,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            **self._predict_kwargs(contexts, cross_learning=False),
        )
        return np.stack(
            [self._to_numpy(mean)[0, :horizon].astype(np.float32) for mean in means]
        )

    def _build_context(self, series: pd.DataFrame) -> np.ndarray:
        """Legacy single-series helper used by fine-tuning and existing tests."""
        values = self._series_to_tensor(series).squeeze()
        if values.ndim == 2:
            values = values[:, 0]
        return values.astype(np.float32)

    def _build_inputs(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        series: pd.DataFrame,
        covariates: Optional[dict] = None,
        horizon: Optional[int] = None,
        **input_meta: Any,
    ) -> Tuple[list, _InputLayout]:
        """Convert wide or grouped repository data into Chronos list inputs."""
        grouped = self._is_grouped(series, input_meta)
        if grouped:
            targets, layout = self._grouped_targets(series, input_meta)
        else:
            targets, layout = self._wide_targets(series)

        if not self._use_covariates or not covariates or horizon is None:
            return [target for target, _ in targets], layout

        items = []
        for item_idx, (target, history_index) in enumerate(targets):
            group_id = layout.item_ids[item_idx] if grouped else None
            past, future = self._item_covariates(
                covariates,
                history_index,
                horizon,
                group_id=group_id,
                group_col=layout.group_col,
                time_col=layout.time_col,
            )
            item: Dict[str, Any] = {"target": target}
            if past:
                item["past_covariates"] = past
            if future:
                item["future_covariates"] = future
            items.append(item)

        self._harmonize_covariate_schema(items)
        if all(set(item) == {"target"} for item in items):
            return [item["target"] for item in items], layout
        return items, layout

    @staticmethod
    def _is_grouped(series: pd.DataFrame, meta: Dict[str, Any]) -> bool:
        group_col = meta.get("group_col")
        time_col = meta.get("time_col")
        return bool(
            meta.get("pipeline_type") == "static_grouped"
            and group_col in series.columns
            and time_col in series.columns
        )

    def _wide_targets(
        self, series: pd.DataFrame
    ) -> Tuple[List[Tuple[np.ndarray, pd.Index]], _InputLayout]:
        numeric = series.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError("Chronos requires at least one numeric target series.")
        targets: List[Tuple[np.ndarray, pd.Index]] = []
        for name in numeric.columns:
            values, index = self._trim_target(numeric[name], numeric.index)
            targets.append((values, index))
        return targets, _InputLayout(
            grouped=False,
            item_ids=list(numeric.columns),
            target_names=list(numeric.columns),
            history_indices=[idx for _, idx in targets],
        )

    def _grouped_targets(
        self, series: pd.DataFrame, meta: Dict[str, Any]
    ) -> Tuple[List[Tuple[np.ndarray, pd.Index]], _InputLayout]:
        group_col = str(meta["group_col"])
        time_col = str(meta["time_col"])
        target_cols = [c for c in (meta.get("target_cols") or []) if c in series]
        if not target_cols:
            target_cols = [c for c in series if c not in {group_col, time_col}]
        if len(target_cols) != 1:
            raise ValueError(
                "Chronos static_grouped integration currently supports exactly "
                f"one target column; got {target_cols}."
            )
        target_col = target_cols[0]
        if not pd.api.types.is_numeric_dtype(series[target_col]):
            raise ValueError(f"Chronos target {target_col!r} must be numeric.")

        targets: List[Tuple[np.ndarray, pd.Index]] = []
        item_ids: List[Any] = []
        for group_id, group in series.groupby(group_col, sort=False):
            ordered = group.sort_values(time_col)
            time_index = pd.Index(pd.to_datetime(ordered[time_col]))
            values, index = self._trim_target(ordered[target_col], time_index)
            targets.append((values, index))
            item_ids.append(group_id)
        return targets, _InputLayout(
            grouped=True,
            item_ids=item_ids,
            target_names=[target_col] * len(item_ids),
            history_indices=[idx for _, idx in targets],
            group_col=group_col,
            time_col=time_col,
        )

    @staticmethod
    def _trim_target(values: pd.Series, index: pd.Index) -> Tuple[np.ndarray, pd.Index]:
        valid = values.notna().to_numpy()
        if not valid.any():
            raise ValueError(f"Target series {values.name!r} contains no observations.")
        positions = np.flatnonzero(valid)
        start, end = int(positions[0]), int(positions[-1]) + 1
        target = values.iloc[start:end].to_numpy(dtype=np.float32)
        return target, index[start:end]

    def _item_covariates(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        covariates: dict,
        history_index: pd.Index,
        horizon: int,
        *,
        group_id: Any,
        group_col: Optional[str],
        time_col: Optional[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        past: Dict[str, np.ndarray] = {}
        future: Dict[str, np.ndarray] = {}

        past_frame = self._select_covariate_frame(
            covariates.get("past_covariates"), group_id, group_col, time_col
        )
        for name in past_frame.columns:
            past[str(name)] = self._align_history(past_frame[name], history_index)

        future_frame = self._select_covariate_frame(
            covariates.get("future_covariates"), group_id, group_col, time_col
        )
        if not future_frame.empty:
            future_index = self._future_index(history_index, horizon)
            for name in future_frame.columns:
                key = str(name)
                past[key] = self._align_history(future_frame[name], history_index)
                future[key] = self._align_future(
                    future_frame[name], history_index, future_index, horizon
                )

        static_frame = self._select_covariate_frame(
            covariates.get("static_covariates"), group_id, group_col, time_col
        )
        if not static_frame.empty:
            for name in static_frame.columns:
                value = static_frame[name].iloc[0]
                dtype = object if isinstance(value, str) else None
                past[str(name)] = np.asarray([value] * len(history_index), dtype=dtype)
                future[str(name)] = np.asarray([value] * horizon, dtype=dtype)
        return past, future

    @staticmethod
    def _select_covariate_frame(
        frame: Any,
        group_id: Any,
        group_col: Optional[str],
        time_col: Optional[str],
    ) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        result = frame.copy()
        if group_col and group_col in result.columns and group_id is not None:
            result = result[result[group_col] == group_id]
        if time_col and time_col in result.columns:
            result[time_col] = pd.to_datetime(result[time_col])
            result = result.sort_values(time_col).set_index(time_col)
        result = result.drop(
            columns=[c for c in (group_col, time_col) if c], errors="ignore"
        )
        return result

    @staticmethod
    def _align_history(column: pd.Series, history_index: pd.Index) -> np.ndarray:
        if isinstance(column.index, pd.DatetimeIndex) and isinstance(
            history_index, pd.DatetimeIndex
        ):
            values = column.reindex(history_index).to_numpy()
        else:
            values = column.to_numpy()[: len(history_index)]
            if len(values) < len(history_index):
                pad = np.full(len(history_index) - len(values), np.nan, dtype=object)
                values = np.concatenate([pad, values])
        return ChronosAdapter._coerce_covariate(values)

    @staticmethod
    def _align_future(
        column: pd.Series,
        history_index: pd.Index,
        future_index: pd.Index,
        horizon: int,
    ) -> np.ndarray:
        if isinstance(column.index, pd.DatetimeIndex) and isinstance(
            future_index, pd.DatetimeIndex
        ):
            values = column.reindex(future_index).to_numpy()
        else:
            start = len(history_index)
            values = column.to_numpy()[start : start + horizon]
            if len(values) < horizon:
                values = np.pad(
                    values, (0, horizon - len(values)), constant_values=np.nan
                )
        return ChronosAdapter._coerce_covariate(values)

    @staticmethod
    def _coerce_covariate(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(np.float32)
        return arr.astype(object)

    @staticmethod
    def _harmonize_covariate_schema(items: List[Dict[str, Any]]) -> None:
        past_keys = sorted(
            {key for item in items for key in item.get("past_covariates", {})}
        )
        future_keys = sorted(
            {key for item in items for key in item.get("future_covariates", {})}
        )
        for item in items:
            history_len = len(item["target"])
            past = item.setdefault("past_covariates", {})
            future = item.setdefault("future_covariates", {})
            for key in past_keys:
                past.setdefault(key, np.full(history_len, np.nan, dtype=np.float32))
            for key in future_keys:
                if key not in future:
                    example = next(
                        candidate["future_covariates"][key]
                        for candidate in items
                        if key in candidate.get("future_covariates", {})
                    )
                    future[key] = np.full(len(example), np.nan, dtype=np.float32)
            if not past:
                item.pop("past_covariates", None)
            if not future:
                item.pop("future_covariates", None)

    def _predict_kwargs(self, inputs: list, *, cross_learning: bool) -> dict:
        if cross_learning and len(inputs) > 1:
            required = sum(
                1 + len(item.get("past_covariates", {}))
                if isinstance(item, dict)
                else 1
                for item in inputs
            )
            if self._batch_size < required:
                raise ValueError(
                    "Chronos cross-learning group would be split by batch_size: "
                    f"requires at least {required}, got {self._batch_size}."
                )
        kwargs = {
            "batch_size": self._batch_size,
            "cross_learning": cross_learning,
        }
        if self._context_length is not None:
            kwargs["context_length"] = self._context_length
        return kwargs

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _stack_means(cls, means: list, horizon: int) -> np.ndarray:
        return np.column_stack(
            [cls._to_numpy(mean)[0, :horizon] for mean in means]
        ).astype(np.float32)

    @classmethod
    def _stack_quantiles(cls, quantiles: list, horizon: int) -> np.ndarray:
        return np.stack(
            [cls._to_numpy(item)[0, :horizon, :] for item in quantiles], axis=1
        ).astype(np.float32)

    @staticmethod
    def _stack_distributions(
        forecasts: List[DistributionForecast],
    ) -> DistributionForecast:
        name = forecasts[0].distribution_name
        params = {
            key: np.column_stack([forecast.params[key] for forecast in forecasts])
            for key in forecasts[0].params
        }
        return DistributionForecast(name, params)

    def _grouped_point_frame(
        self, point: np.ndarray, layout: _InputLayout, horizon: int
    ) -> pd.DataFrame:
        assert layout.group_col is not None and layout.time_col is not None
        target_col = layout.target_names[0]
        frames = []
        for item_idx, item_id in enumerate(layout.item_ids):
            index = self._future_index(layout.history_indices[item_idx], horizon)
            frames.append(
                pd.DataFrame(
                    {
                        layout.time_col: index,
                        layout.group_col: item_id,
                        target_col: point[:, item_idx],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _future_index(history_index: pd.Index, horizon: int) -> pd.Index:
        if isinstance(history_index, pd.DatetimeIndex):
            freq = pd.infer_freq(history_index) if len(history_index) >= 3 else None
            if freq is None and len(history_index) >= 2:
                freq = history_index[-1] - history_index[-2]
            if freq is None:
                freq = pd.Timedelta(1, "D")
            return pd.date_range(history_index[-1], periods=horizon + 1, freq=freq)[1:]
        if len(history_index) >= 2:
            step = history_index[-1] - history_index[-2]
        else:
            step = 1
        start = history_index[-1] + step
        return pd.Index([start + step * i for i in range(horizon)])
