from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.schema.foundation_config import FineTuneConfig

try:
    from momentfm import MOMENTPipeline

    _MOMENT_AVAILABLE = True
except ImportError:
    _MOMENT_AVAILABLE = False


class MomentAdapter(FoundationModelBase):  # pylint: disable=abstract-method
    """
    Adapter for MOMENT (CMU AutonLab) foundation model.

    Supports zero-shot and fine-tuned forecasting.
    MOMENT also supports anomaly detection and classification via
    task_name — this adapter focuses on the forecasting task.

    Required extra dependency:
        pip install momentfm
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "AutonLab/MOMENT-1-large",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        context_length: int = 512,
        torch_dtype: str = "bfloat16",
        finetune_config: Optional[dict] = None,
        **_kwargs,
    ):
        super().__init__(
            pretrained_model_id=pretrained_model_id,
            inference_mode=inference_mode,
            device=device,
            torch_dtype=torch_dtype,
            context_length=context_length,
            finetune_config=finetune_config,
        )
        self._head: Any = None

    @property
    def model_name(self) -> str:
        return f"moment[{self._model_id}]"

    @property
    def supports_finetuning(self) -> bool:
        return True

    @property
    def supports_probabilistic(self) -> bool:
        return True

    def _enable_mc_dropout(self) -> None:
        """Set model to eval but keep Dropout layers active for MC uncertainty."""
        if self._model is None:  # type: ignore[has-type]
            return
        self._model.eval()  # type: ignore[has-type]
        for m in self._model.modules():  # type: ignore[has-type]
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def _load_model(self, horizon: int = 1) -> None:
        if self._model is not None:  # type: ignore[has-type]
            return
        if not _MOMENT_AVAILABLE:
            raise ImportError(
                "momentfm is required for MomentAdapter.\n"
                "Install with: pip install momentfm"
            )

        torch_dtype = self._resolve_torch_dtype()

        self._model = MOMENTPipeline.from_pretrained(
            self._model_id,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": horizon,
                "head_dropout": 0.1,
                "weight_decay": 0.0,
                "freeze_encoder": True,
                "freeze_embedder": True,
            },
            torch_dtype=torch_dtype,
        )
        self._model.init()  # type: ignore[attr-defined]
        self._model = self._model.to(self._device)  # type: ignore[attr-defined]

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        self._load_model(horizon=horizon)
        context = self._prepare_context(series)
        with torch.no_grad():
            output = self._model(x_enc=context)
        preds = output.forecast.squeeze().cpu().numpy()
        # squeeze → (n_channels, horizon); transpose to (horizon, n_channels)
        if preds.ndim == 2:
            preds = preds.T  # (n_channels, horizon) → (horizon, n_channels)
        return preds[:horizon].astype(np.float32)

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """MC Dropout: run num_samples forward passes with dropout enabled.

        MOMENT uses head_dropout=0.1, giving per-sample uncertainty from the
        forecasting head. The encoder is frozen so variance comes from the head.
        """
        self._load_model(horizon=horizon)
        context = self._prepare_context(series)
        self._enable_mc_dropout()
        draws: list = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = self._model(x_enc=context)
                pred = output.forecast.squeeze().cpu().numpy()
                if pred.ndim == 2:
                    pred = pred.T  # (n_channels, horizon) → (horizon, n_channels)
                draws.append(pred[:horizon])
        self._model.eval()
        return np.stack(draws).astype(np.float32)  # (num_samples, horizon)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> DistributionForecast:
        """Fit a parametric distribution to MC Dropout samples."""
        samples = self.forecast_probabilistic(
            horizon, series, num_samples=num_samples, covariates=covariates
        )
        return DistributionForecast.from_samples(samples, distribution_name)

    def _build_windows(self, values, ctx_len, horizon, stride):
        """Slide a window over values and return (x_windows, y_windows) lists."""
        x_wins, y_wins = [], []
        n_timesteps = len(values)
        for i in range(0, n_timesteps - ctx_len - horizon + 1, stride):
            x_wins.append(values[i : i + ctx_len])
            y_wins.append(values[i + ctx_len : i + ctx_len + horizon])
        return x_wins, y_wins

    def _train_head(self, x_t, y_t, config):
        """Train the forecasting head (encoder frozen) on pre-built window tensors."""
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        criterion = torch.nn.MSELoss()
        n_windows = len(x_t)
        batch_size = min(config.batch_size, n_windows)
        self._model.train()
        for _ in range(config.epochs):
            perm = torch.randperm(n_windows, device=self._device)
            for start in range(0, n_windows, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                out = self._model(x_enc=x_t[idx])
                loss = criterion(out.forecast, y_t[idx])
                loss.backward()
                optimizer.step()
        self._model.eval()

    def finetune(
        self,
        train_data: pd.DataFrame,
        config: "FineTuneConfig",
        val_data: Optional[pd.DataFrame] = None,
    ) -> "MomentAdapter":
        """Linear probing: freeze encoder + embedder, train only the forecasting head.

        Builds sliding windows (context=512, target=forecast_horizon) from train_data
        and trains the head via mini-batch SGD. Matches the MOMENT_LP paper setup.
        """
        horizon = config.forecast_horizon
        self._load_model(horizon=horizon)

        ctx_len = self._context_length or 512
        values = self._series_to_tensor(train_data)  # (n_timesteps, n_channels)
        x_wins, y_wins = self._build_windows(values, ctx_len, horizon, config.stride)

        if not x_wins:
            return self

        # (N, n_channels, ctx_len) and (N, n_channels, horizon)
        x_t = (
            torch.tensor(np.stack(x_wins), dtype=torch.float32)
            .permute(0, 2, 1)
            .to(self._device)
        )
        y_t = (
            torch.tensor(np.stack(y_wins), dtype=torch.float32)
            .permute(0, 2, 1)
            .to(self._device)
        )

        self._train_head(x_t, y_t, config)
        return self

    def _prepare_context(self, series: pd.DataFrame) -> "torch.Tensor":
        values = self._series_to_tensor(series)
        # MOMENT expects (batch, n_channels, seq_len)
        context_length = self._context_length
        ctx = values if context_length is None else values[-context_length:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).T.unsqueeze(0)
        return ctx_tensor.to(self._device)
