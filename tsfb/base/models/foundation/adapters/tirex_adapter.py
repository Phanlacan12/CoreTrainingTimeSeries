"""TiRex-2 zero-shot adapter with optional dynamic covariates."""

# Adapter interfaces intentionally mirror the other foundation adapters.
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.forecasting.quantile_utils import fit_distribution_from_quantiles
from tsfb.base.models.foundation.base_foundation import FoundationModelBase

try:
    import tirex2
    from tirex2.model.types import TimeseriesType

    _TIREX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the optional local extra
    tirex2 = None  # type: ignore[assignment]
    TimeseriesType = None  # type: ignore[assignment,misc]
    _TIREX_AVAILABLE = False


class TiRexAdapter(FoundationModelBase):  # pylint: disable=abstract-method
    """Adapt one repository target series to the TiRex-2 inference API.

    Multiple repository targets are intentionally handled by
    ``UnivariateToMultivariate``. TiRex's list API batches independent items;
    it does not provide the cross-learning switch exposed by Chronos-2.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "NX-AI/TiRex-2",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        num_samples: int = 20,
        context_length: Optional[int] = None,
        use_covariates: bool = False,
        cross_learning: bool = False,
        **_kwargs: Any,
    ) -> None:
        if inference_mode != "zeroshot":
            raise ValueError(
                "TiRex-2 currently supports only inference_mode='zeroshot'."
            )
        if context_length is not None:
            raise ValueError(
                "TiRex-2 does not expose context_length through its forecast API. "
                "Provide the actual available history in the input data instead."
            )
        if cross_learning:
            raise ValueError(
                "TiRex-2 has no cross_learning option. Use "
                "approach='univariate_per_series' for independent targets."
            )

        super().__init__(
            pretrained_model_id=pretrained_model_id,
            device=device,
            inference_mode=inference_mode,
            context_length=None,
        )
        self._num_samples = int(num_samples)
        self._use_covariates = bool(use_covariates)
        self._pipeline: Any = None
        self._timeseries_type = TimeseriesType

    @property
    def model_name(self) -> str:
        return f"tirex[{self._model_id}]"

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
        if not _TIREX_AVAILABLE or tirex2 is None:
            raise ImportError(
                "tirex-2 is required for TiRexAdapter. Install the optional "
                "stack with: python -m pip install --no-deps tirex-2==0.1.1 "
                "xlstm==2.0.5 flashrnn==1.0.6 mlstm-kernels==2.0.4"
            )
        self._pipeline = tirex2.load_model(self._model_id, device=self._device)

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        **_input_meta: Any,
    ) -> np.ndarray:
        quantiles, levels = self._predict_quantiles(horizon, series, covariates)
        median_idx = self._quantile_index(levels, 0.5)
        return quantiles[median_idx].astype(np.float32)

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,
        **_input_meta: Any,
    ) -> np.ndarray:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        quantiles, levels = self._predict_quantiles(horizon, series, covariates)
        monotone = np.maximum.accumulate(quantiles, axis=0)
        probabilities = (np.arange(num_samples, dtype=float) + 0.5) / num_samples
        samples = np.column_stack(
            [
                np.interp(probabilities, levels, monotone[:, step])
                for step in range(horizon)
            ]
        )
        return samples.astype(np.float32)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
        **_input_meta: Any,
    ) -> DistributionForecast:
        quantiles, levels = self._predict_quantiles(horizon, series, covariates)
        if distribution_name == "negative_binomial":
            samples = self._samples_from_quantiles(quantiles, levels, num_samples)
            return DistributionForecast.from_samples(samples, distribution_name)

        params = fit_distribution_from_quantiles(
            quantiles.T, levels.tolist(), distribution_name
        )
        return DistributionForecast(distribution_name, params)

    def _predict_quantiles(
        self,
        horizon: int,
        series: pd.DataFrame,
        covariates: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._load_pipeline()
        self._validate_horizon(horizon)
        target, history_index = self._target_context(series)
        past, future = self._covariate_tensors(
            covariates, history_index, horizon
        )
        item = self._make_timeseries(target, past, future)
        outputs = self._pipeline.forecast(
            [item], prediction_length=horizon, output_type="numpy"
        )
        levels = self._quantile_levels()
        quantiles = self._extract_output(outputs, levels, horizon)
        return quantiles, levels

    def _validate_horizon(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        maximum = int(getattr(self._pipeline, "future_len", 0) or 0)
        if maximum and horizon > maximum:
            raise ValueError(
                f"TiRex-2 supports horizon <= {maximum}; got {horizon}."
            )

    @staticmethod
    def _target_context(series: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        if not isinstance(series, pd.DataFrame) or series.shape[1] != 1:
            width = series.shape[1] if isinstance(series, pd.DataFrame) else "unknown"
            raise ValueError(
                "TiRexAdapter requires exactly one target column per call; "
                f"got {width}. Use approach='univariate_per_series'."
            )
        target = series.iloc[:, 0]
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(f"TiRex target {target.name!r} must be numeric.")
        valid_positions = np.flatnonzero(target.notna().to_numpy())
        if len(valid_positions) == 0:
            raise ValueError(f"Target series {target.name!r} contains no observations.")
        start = int(valid_positions[0])
        observed = target.iloc[start:]
        if observed.isna().any():
            raise ValueError(
                f"Target series {target.name!r} may contain missing values only "
                "before its first observation."
            )
        return observed.to_numpy(dtype=np.float32), observed.index

    def _covariate_tensors(
        self,
        covariates: Optional[dict],
        history_index: pd.Index,
        horizon: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._use_covariates:
            return None, None
        if not covariates:
            raise ValueError(
                "use_covariates=True but no past or future covariates were provided."
            )
        static = covariates.get("static_covariates")
        if isinstance(static, pd.DataFrame) and not static.empty:
            raise ValueError("TiRex-2 does not support static_covariates.")

        past = self._align_covariates(
            covariates.get("past_covariates"), history_index, "past_covariates"
        )
        future_index = self._future_index(history_index, horizon)
        combined_index = history_index.append(future_index)
        future = self._align_covariates(
            covariates.get("future_covariates"),
            combined_index,
            "future_covariates",
        )
        if past is None and future is None:
            raise ValueError(
                "use_covariates=True but no supported dynamic covariates were provided."
            )
        return past, future

    @staticmethod
    def _align_covariates(
        frame: Any, required_index: pd.Index, name: str
    ) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty pandas DataFrame.")
        if frame.index.has_duplicates:
            raise ValueError(f"{name} index contains duplicate timestamps.")
        non_numeric = [
            str(column)
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise ValueError(
                f"TiRex-2 supports only numeric {name}; got {non_numeric}."
            )

        aligned = frame.reindex(required_index)
        if aligned.isna().any().any():
            missing = int(aligned.isna().any(axis=1).sum())
            raise ValueError(
                f"{name} does not cover the required target history/horizon; "
                f"{missing} aligned rows contain missing values."
            )
        return aligned.to_numpy(dtype=np.float32).T

    def _make_timeseries(
        self,
        target: np.ndarray,
        past: Optional[np.ndarray],
        future: Optional[np.ndarray],
    ) -> Any:
        if self._timeseries_type is None:
            raise ImportError("TiRex TimeseriesType is unavailable.")
        return self._timeseries_type(
            target=torch.from_numpy(target[np.newaxis, :]),
            past_covariates=None if past is None else torch.from_numpy(past),
            future_covariates=None if future is None else torch.from_numpy(future),
        )

    def _quantile_levels(self) -> np.ndarray:
        getter = getattr(self._pipeline, "_quantile_levels", None)
        if callable(getter):
            raw = getter()  # pylint: disable=not-callable
        else:
            raw = getattr(self._pipeline, "quantiles")
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        levels = np.asarray(raw, dtype=float).reshape(-1)
        if len(levels) == 0 or not np.all(np.diff(levels) > 0):
            raise ValueError("TiRex-2 returned an invalid quantile grid.")
        self._quantile_index(levels, 0.5)
        return levels

    @staticmethod
    def _quantile_index(levels: np.ndarray, level: float) -> int:
        matches = np.flatnonzero(np.isclose(levels, level))
        if len(matches) != 1:
            raise ValueError(f"TiRex-2 quantile grid must contain q={level}.")
        return int(matches[0])

    @staticmethod
    def _extract_output(
        outputs: Any, levels: np.ndarray, horizon: int
    ) -> np.ndarray:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ValueError("TiRex-2 must return one output for one input item.")
        result = np.asarray(outputs[0])
        expected = (1, len(levels), horizon)
        if result.shape != expected:
            raise ValueError(
                f"Unexpected TiRex-2 output shape {result.shape}; expected {expected}."
            )
        if not np.isfinite(result).all():
            raise ValueError("TiRex-2 returned non-finite forecast values.")
        return result[0].astype(np.float32)

    @staticmethod
    def _samples_from_quantiles(
        quantiles: np.ndarray, levels: np.ndarray, num_samples: int
    ) -> np.ndarray:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        monotone = np.maximum.accumulate(quantiles, axis=0)
        probabilities = (np.arange(num_samples, dtype=float) + 0.5) / num_samples
        return np.column_stack(
            [
                np.interp(probabilities, levels, monotone[:, step])
                for step in range(quantiles.shape[1])
            ]
        ).astype(np.float32)

    @staticmethod
    def _future_index(history_index: pd.Index, horizon: int) -> pd.Index:
        if len(history_index) == 0:
            raise ValueError("Target history must not be empty.")
        if isinstance(history_index, pd.DatetimeIndex):
            freq = pd.infer_freq(history_index) if len(history_index) >= 3 else None
            if freq is None and len(history_index) >= 2:
                freq = history_index[-1] - history_index[-2]
            if freq is None:
                raise ValueError("Cannot infer target frequency from one timestamp.")
            return pd.date_range(
                history_index[-1], periods=horizon + 1, freq=freq
            )[1:]
        step = history_index[-1] - history_index[-2] if len(history_index) >= 2 else 1
        return pd.Index([history_index[-1] + step * (i + 1) for i in range(horizon)])
