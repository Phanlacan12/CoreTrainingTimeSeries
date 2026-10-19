from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.schema.foundation_config import FineTuneConfig

try:
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiForecast, MoiraiModule

    _MOIRAI_AVAILABLE = True
except ImportError:
    _MOIRAI_AVAILABLE = False


class MoiraiAdapter(FoundationModelBase):  # pylint: disable=abstract-method
    """
    Adapter for Moirai (Salesforce) universal time series foundation model.

    Supports zero-shot, probabilistic forecasting, and fine-tuning.

    Required extra dependency:
        pip install uni2ts
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "Salesforce/moirai-1.0-R-small",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        num_samples: int = 20,
        context_length: int = 512,
        patch_size: str = "auto",
        torch_dtype: str = "bfloat16",
        finetune_config: Optional[dict] = None,
        **_kwargs,
    ):
        super().__init__(
            device=device,
            pretrained_model_id=pretrained_model_id,
            context_length=context_length,
            inference_mode=inference_mode,
            finetune_config=finetune_config,
            torch_dtype=torch_dtype,
        )
        self._num_samples = num_samples
        self._patch_size = patch_size
        self._pipeline: Any = None

    @property
    def model_name(self) -> str:
        return f"moirai[{self._model_id}]"

    @property
    def supports_finetuning(self) -> bool:
        return True

    @property
    def supports_probabilistic(self) -> bool:
        return True

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        if not _MOIRAI_AVAILABLE:
            raise ImportError(
                "uni2ts is required for MoiraiAdapter.\n"
                "Install with: pip install uni2ts"
            )
        self._pipeline = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self._model_id),
            prediction_length=1,
            context_length=self._context_length,
            patch_size=self._patch_size,
            num_samples=self._num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._pipeline = self._pipeline.to(self._device)

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        samples = self._predict_samples(horizon, series)
        return np.median(samples, axis=0)

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        return self._predict_samples(horizon, series, num_samples=num_samples)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> DistributionForecast:
        """Fit a parametric distribution to Moirai's Monte Carlo sample output."""
        samples = self._predict_samples(horizon, series, num_samples=num_samples)
        return DistributionForecast.from_samples(samples, distribution_name)

    def finetune(
        self,
        train_data: pd.DataFrame,
        config: "FineTuneConfig",
        val_data: Optional[pd.DataFrame] = None,
    ) -> "MoiraiAdapter":
        self._load_pipeline()
        if not _MOIRAI_AVAILABLE:
            raise ImportError("uni2ts is required for Moirai fine-tuning")

        context = self._series_to_tensor(train_data)
        # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        finetune_module = MoiraiFinetune(
            module=self._pipeline.module,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            finetune_mode=config.mode,
            lora_rank=config.lora_rank if config.mode == "lora" else None,
        )
        finetune_module.fit(
            inputs=[context.squeeze()],
            num_epochs=config.epochs,
            batch_size=config.batch_size,
        )
        self._pipeline.module = finetune_module.module
        return self

    def _predict_samples(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        self._load_pipeline()
        n = num_samples or self._num_samples
        context = (
            torch.tensor(
                self._series_to_tensor(series).squeeze(),
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # (1, T, 1)

        self._pipeline.prediction_length = horizon
        self._pipeline.num_samples = n

        with torch.no_grad():
            forecast = self._pipeline(context)

        samples = forecast.samples.squeeze(-1).cpu().numpy()  # (num_samples, horizon)
        return samples
