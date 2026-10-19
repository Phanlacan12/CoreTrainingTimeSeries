from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from tsfb.base.data_module.preprocess_model.tfb_grouped_preprocess import (
    TFBGroupedPreprocess,
)
from tsfb.base.data_module.preprocess_model.tfb_preprocess import (
    DeepWindowPreprocess,
    TFBPreprocess,
)
from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.base_model import BatchMaker, ModelBase
from tsfb.base.models.tfb.config_builder import build_config_namespace
from tsfb.base.models.tfb.registry import get_model_spec
from tsfb.base.models.tfb.trainer import (
    TrainingConfig,
    _align_prediction,
    _forward_batch,
    fit_model,
    predict_batches,
)


class TFBModelAdapter(ModelBase):
    """Adapter that wraps origin_code models under the benchmark ModelBase API."""

    def __init__(self, model_key: str, **model_hyper_params: Any):
        self.spec = get_model_spec(model_key)
        self.model_hyper_params = dict(model_hyper_params)
        self._model: Optional[nn.Module] = None
        self._preprocess: Optional[DeepWindowPreprocess] = None
        self._config: Optional[SimpleNamespace] = None
        self._target_cols: List[str] = []
        self._group_col: Optional[str] = None
        self._time_col: Optional[str] = None
        self._device = self.model_hyper_params.get(
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

    @property
    def model_name(self) -> str:
        return f"tfb.{self.spec.public_name}"

    def _build_preprocess(
        self,
        train_data: pd.DataFrame,
        **kwargs: Any,
    ) -> DeepWindowPreprocess:
        grouped = (
            kwargs.get("group_col") in train_data.columns
            and kwargs.get("time_col") in train_data.columns
        )
        if grouped:
            target_cols = list(kwargs.get("target_cols") or [])
            if not target_cols:
                target_cols = [
                    col
                    for col in train_data.columns
                    if col not in {kwargs["group_col"], kwargs["time_col"]}
                ]
            self._target_cols = target_cols
            self._group_col = kwargs["group_col"]
            self._time_col = kwargs["time_col"]
        else:
            self._target_cols = list(train_data.columns)
            self._group_col = None
            self._time_col = None

        preprocess_cfg = {
            "family": self.spec.family.value,
            "seq_len": int(self.model_hyper_params["input_chunk_length"]),
            "pred_len": int(self.model_hyper_params["output_chunk_length"]),
            "label_len": int(
                self.model_hyper_params.get(
                    "label_len",
                    max(1, int(self.model_hyper_params["input_chunk_length"]) // 2),
                )
            ),
            "batch_size": int(self.model_hyper_params.get("batch_size", 32)),
            "num_workers": int(self.model_hyper_params.get("num_workers", 0)),
            "freq": self.model_hyper_params.get("freq", kwargs.get("freq", "D")),
            "target_columns": self._target_cols,
        }
        if grouped:
            preprocess_cfg.update(
                {"group_col": self._group_col, "time_col": self._time_col}
            )
            return TFBGroupedPreprocess(preprocess_cfg)
        return TFBPreprocess(preprocess_cfg)

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[Dict[str, Any]] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs: Any,
    ) -> "TFBModelAdapter":
        self._preprocess = self._build_preprocess(train_data, **kwargs)
        self._preprocess.fit(
            train_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        self._config = build_config_namespace(
            self.spec,
            self.model_hyper_params,
            num_target_channels=len(self._target_cols),
            data_freq=self.model_hyper_params.get("freq", kwargs.get("freq", "D")),
        )
        model_cls = self.spec.load_class()
        model = model_cls(self._config)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Loaded model for {self.spec.public_name} must be a torch.nn.Module."
            )
        self._model = model
        fit_model(
            model,
            train_loader=self._preprocess.build_train_loader(),
            val_loader=self._preprocess.build_val_loader(),
            family=self.spec.family,
            config=TrainingConfig(
                n_epochs=int(self.model_hyper_params.get("n_epochs", 1)),
                learning_rate=float(self.model_hyper_params.get("learning_rate", 1e-3)),
                weight_decay=float(self.model_hyper_params.get("weight_decay", 0.0)),
                patience=self.model_hyper_params.get("patience"),
                device=str(self._device),
            ),
            loss_name=str(self.model_hyper_params.get("loss_name", "mse")),
        )
        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self._model is None or self._preprocess is None:
            raise RuntimeError("Call forecast_fit() before forecast().")
        expected = int(self.model_hyper_params["output_chunk_length"])
        if horizon != expected:
            raise ValueError(f"Expected horizon={expected}, got {horizon}.")

        inference_payload = self._preprocess.predict(series, covariates=covariates)
        preds = predict_batches(
            self._model,
            loader=self._preprocess.build_inference_loader(),
            family=self.spec.family,
            device=str(self._device),
        ).numpy()

        if self._group_col and self._time_col:
            return self._preprocess.inverse_predict(
                self._format_grouped_predictions(inference_payload, preds)
            )
        return self._preprocess.inverse_predict(preds[:, :horizon, :][0])

    def _format_grouped_predictions(
        self,
        inference_payload: Dict[str, Any],
        preds: np.ndarray,
    ) -> pd.DataFrame:
        samples = inference_payload["samples"]
        prediction_index = samples["prediction_index"]
        group_ids = samples["group_id"]
        rows = []
        for sample_idx, (gid, pred_index) in enumerate(
            zip(group_ids, prediction_index)
        ):
            pred_values = preds[sample_idx]
            df_pred = pd.DataFrame(pred_values, columns=self._target_cols)
            df_pred[self._time_col] = pred_index
            df_pred[self._group_col] = gid
            rows.append(df_pred[[self._time_col, self._group_col] + self._target_cols])
        return (
            pd.concat(rows, ignore_index=True)
            .sort_values([self._group_col, self._time_col])
            .reset_index(drop=True)
        )

    @property
    def supports_probabilistic(self) -> bool:
        """True when the model has been fitted and contains Dropout layers."""
        if self._model is None:
            return False
        return any(isinstance(m, nn.Dropout) for m in self._model.modules())

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """MC Dropout: run num_samples stochastic forward passes.

        Keeps Dropout layers active (train mode) while disabling gradient
        computation. Returns None when no Dropout layers are present.
        Returns shape (num_samples, horizon) for univariate series,
        or (num_samples, horizon, n_targets) for multivariate.
        """
        if self._model is None or self._preprocess is None:
            raise RuntimeError("Call forecast_fit() before forecast_probabilistic().")
        if not self.supports_probabilistic:
            return None

        expected = int(self.model_hyper_params["output_chunk_length"])
        if horizon != expected:
            raise ValueError(f"Expected horizon={expected}, got {horizon}.")

        # Build inference state once; DataLoader is re-iterable
        self._preprocess.predict(series, covariates=covariates)
        loader = self._preprocess.build_inference_loader()
        torch_device = torch.device(str(self._device))
        self._model.to(torch_device).eval()

        # Enable only Dropout layers for MC uncertainty
        for m in self._model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        samples: list = []
        with torch.no_grad():
            for _ in range(num_samples):
                batch_preds: list = []
                for batch in loader:
                    batch = {
                        k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    pred = (
                        _align_prediction(
                            _forward_batch(self._model, batch, self.spec.family),
                            batch["y"],
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    batch_preds.append(pred)
                if batch_preds:
                    # shape: (batch, horizon, n_targets) → take first window
                    combined = np.concatenate(batch_preds, axis=0)
                    samples.append(combined[0, :horizon])

        self._model.eval()
        if not samples:
            return np.empty((0, horizon))
        arr = np.stack(samples).astype(np.float32)  # (S, horizon[, T])
        return arr.squeeze(-1) if arr.ndim == 3 and arr.shape[-1] == 1 else arr

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[Dict[str, Any]] = None,
    ) -> Optional[DistributionForecast]:
        """Fit a parametric distribution to MC Dropout samples.

        Returns None when no Dropout layers are present (deterministic model).
        """
        samples = self.forecast_probabilistic(
            horizon, series, num_samples=num_samples, covariates=covariates
        )
        if samples is None:
            return None
        return DistributionForecast.from_samples(samples, distribution_name)

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        **kwargs: Any,
    ) -> np.ndarray:
        """Expose the batch-forecast interface explicitly for TFB adapters."""
        raise NotImplementedError(
            "TFBModelAdapter does not yet implement batch_forecast()."
        )
