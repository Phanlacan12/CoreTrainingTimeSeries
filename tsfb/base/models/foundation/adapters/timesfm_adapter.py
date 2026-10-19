from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.forecasting.quantile_utils import fit_distribution_from_quantiles
from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.schema.foundation_config import FineTuneConfig

try:
    import timesfm

    _TIMESFM_AVAILABLE = True
except ImportError:
    _TIMESFM_AVAILABLE = False

# Default quantile levels: channels 1-9 of TimesFM 2.x output.
# Layout: channel 0 = extra, channels 1-9 = Q0.1..Q0.9, channel 5 = Q0.5 (point).
_TIMESFM_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# max_horizon passed to ForecastConfig.  Must be a multiple of output_patch_len=128.
_TIMESFM_DEFAULT_HORIZON = 128


class TimesFMAdapter(FoundationModelBase):  # pylint: disable=abstract-method
    """
    Adapter for TimesFM 2.x (Google) foundation model.

    TimesFM 2.x API:
    - Load:    TimesFM_2p5_200M_torch.from_pretrained(repo_id)
    - Compile: model.compile(ForecastConfig(max_context, max_horizon, ...))
    - Infer:   model.forecast(horizon, inputs=[array])
                 → (point[B,H], all_channels[B,H,10])
               channels: 0=extra, 1-9=Q0.1..Q0.9, 5=Q0.5=point forecast

    Required extra dependency:
        pip install "timesfm>=2.0.0"
    """

    supports_batch_forecast: bool = True

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        context_length: int = 512,
        torch_dtype: str = "bfloat16",
        finetune_config: Optional[dict] = None,
        batch_size: int = 32,
        **_kwargs,
    ):
        super().__init__(
            pretrained_model_id,
            device,
            inference_mode,
            context_length,
            torch_dtype,
            finetune_config,
        )
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return f"timesfm[{self._model_id}]"

    @property
    def supports_finetuning(self) -> bool:
        return False  # timesfm 2.x removed fine-tuning API

    @property
    def supports_probabilistic(self) -> bool:
        return True

    @property
    def supports_distribution(self) -> bool:
        return True

    @property
    def output_type(self) -> str:
        return "distribution"

    def _load_model(self) -> None:
        if self._model is not None:  # type: ignore[has-type]
            return
        if not _TIMESFM_AVAILABLE:
            raise ImportError(
                "timesfm>=2.0.0 is required for TimesFMAdapter.\n"
                "Install with: pip install 'timesfm>=2.0.0'"
            )

        # pylint: disable=no-member
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self._model_id,
            torch_compile=False,  # avoids slow torch.compile on CPU
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self._context_length,
                max_horizon=_TIMESFM_DEFAULT_HORIZON,
                per_core_batch_size=1,
            )
        )
        # pylint: enable=no-member

    def _raw_forecast(self, series: pd.DataFrame) -> tuple:
        """Run model.forecast() and return (mean_arr, quantile_arr).

        mean_arr:     (max_horizon,)  — Q0.5 point forecast
        quantile_arr: (max_horizon, 9) — Q0.1..Q0.9
        """
        self._load_model()
        context = self._series_to_tensor(series).squeeze()  # (n_steps,) float32
        # Returns (point[1,H], all_channels[1,H,10])
        # channel 0 = extra; channels 1-9 = Q0.1..Q0.9
        point_full, quant_full = self._model.forecast(  # type: ignore[has-type]
            horizon=_TIMESFM_DEFAULT_HORIZON,
            inputs=[context],
        )
        return (
            np.array(point_full[0], dtype=np.float32),  # (max_horizon,)
            np.array(quant_full[0, :, 1:], dtype=np.float32),  # (max_horizon, 9)
        )

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        if series.shape[1] > 1:
            raise ValueError(
                f"TimesFMAdapter is univariate; got {series.shape[1]} columns. "
                "Use 'approach: univariate' to forecast each column separately."
            )
        mean_arr, _ = self._raw_forecast(series)
        return mean_arr[:horizon]

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Convert TimesFM quantile output to a pseudo-sample matrix.

        Tiles the 9 native quantile levels to approximate num_samples samples.
        """
        _, quant_arr = self._raw_forecast(series)
        q_array = quant_arr[:horizon]  # (horizon, 9)
        q_samples = q_array.T  # (9, horizon)
        reps = max(1, num_samples // len(_TIMESFM_QUANTILE_LEVELS))
        return np.tile(q_samples, (reps, 1))[:num_samples].astype(np.float32)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> DistributionForecast:
        """Fit a parametric distribution directly from TimesFM quantile output."""
        if distribution_name in ("negative_binomial", "poisson"):
            samples = self.forecast_probabilistic(
                horizon, series, num_samples=num_samples, covariates=covariates
            )
            return DistributionForecast.from_samples(samples, distribution_name)

        _, quant_arr = self._raw_forecast(series)
        q_array = quant_arr[:horizon]  # (horizon, 9)
        params = fit_distribution_from_quantiles(
            q_array, _TIMESFM_QUANTILE_LEVELS, distribution_name
        )
        return DistributionForecast(distribution_name, params)

    def _infer_contexts(self, contexts: list, horizon: int) -> np.ndarray:
        """GPU-parallel TimesFM inference on a mini-batch of context arrays.

        Returns shape (B, horizon).
        """
        self._load_model()
        point_full, _ = self._model.forecast(  # type: ignore[has-type]
            horizon=_TIMESFM_DEFAULT_HORIZON,
            inputs=contexts,
        )
        return np.array(point_full, dtype=np.float32)[:, :horizon]  # (B, horizon)

    def finetune(
        self,
        train_data: pd.DataFrame,
        config: "FineTuneConfig",
        val_data: Optional[pd.DataFrame] = None,
    ) -> "TimesFMAdapter":
        raise NotImplementedError(
            "Fine-tuning is not supported in timesfm 2.x. "
            "Use timesfm<2.0 or a custom training loop."
        )
