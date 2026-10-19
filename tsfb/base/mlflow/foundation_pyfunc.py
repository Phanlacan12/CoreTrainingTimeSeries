"""
Custom MLflow PyFunc wrapper for foundation model adapters.

Extends ModelBasePyFunc to log foundation-model-specific metadata:
  - pretrained_model_id
  - inference_mode
  - fine-tuning config params

Also provides helpers for logging HuggingFace checkpoints as artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd

from tsfb.base.forecasting.probabilistic_forecast import ProbabilisticForecast
from tsfb.base.mlflow.pyfunc_model import ModelBasePyFunc
from tsfb.base.models.foundation.base_foundation import FoundationModelBase


class FoundationModelPyfunc(ModelBasePyFunc):
    """
    MLflow PyFunc wrapper for FoundationModelBase adapters.

    Adds foundation-model-specific parameter logging and HuggingFace
    checkpoint artifact handling on top of ModelBasePyFunc.
    """

    def __init__(
        self,
        model_obj: FoundationModelBase,
        horizon: int = 1,
        num_samples: int = 20,
    ):
        super().__init__(model_obj=model_obj, horizon=horizon)
        self._num_samples = num_samples

    model: FoundationModelBase

    def predict(
        self,
        context: Any,
        model_input: Any,
        params: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Return point forecast; also attach probabilistic quantiles when supported."""
        series = model_input
        point = self.model.forecast(horizon=self.horizon, series=series)
        result = pd.DataFrame({"forecast": point})

        if self.model.supports_probabilistic:
            try:
                samples = self.model.forecast_probabilistic(
                    horizon=self.horizon, series=series, num_samples=self._num_samples
                )
                prob = ProbabilisticForecast(samples)
                result["q10"] = prob.quantile(0.1)
                result["q50"] = prob.median
                result["q90"] = prob.quantile(0.9)
            except NotImplementedError:
                pass

        return result


# ---------------------------------------------------------------------------
# Convenience helpers for logging foundation model params + artifacts
# ---------------------------------------------------------------------------


def log_foundation_model_params(adapter: FoundationModelBase) -> None:
    """
    Log foundation-model-specific params into the active MLflow run.

    Reads private attributes that all adapters are expected to expose.
    Uses getattr with defaults to be tolerant of partial implementations.
    """
    params: Dict[str, Any] = {
        "foundation_model_id": getattr(adapter, "_model_id", "unknown"),
        "inference_mode": getattr(adapter, "_inference_mode", "unknown"),
        "supports_zeroshot": adapter.supports_zeroshot,
        "supports_finetuning": adapter.supports_finetuning,
        "supports_probabilistic": adapter.supports_probabilistic,
        "supports_fewshot": adapter.supports_fewshot,
    }

    raw_ft = getattr(adapter, "_raw_finetune_config", None)
    if raw_ft:
        for k, v in raw_ft.items():
            params[f"finetune_{k}"] = v

    mlflow.log_params(params)


def log_foundation_checkpoint(
    checkpoint_dir: str,
    artifact_path: str = "foundation_checkpoint",
) -> None:
    """Log a HuggingFace checkpoint directory as an MLflow artifact."""
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def log_foundation_model(  # pylint: disable=too-many-arguments
    adapter: FoundationModelBase,
    horizon: int,
    *,
    artifact_path: str = "foundation_model",
    registered_model_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    num_samples: int = 20,
) -> mlflow.models.model.ModelInfo:
    """
    Log a FoundationModelBase adapter to MLflow as a PyFunc model.

    Logs params, optional checkpoint artifacts, and the model itself.
    Returns the MLflow ModelInfo for the logged model.
    """
    log_foundation_model_params(adapter)

    if checkpoint_dir:
        log_foundation_checkpoint(checkpoint_dir)

    pyfunc_wrapper = FoundationModelPyfunc(
        model_obj=adapter, horizon=horizon, num_samples=num_samples
    )

    model_info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=pyfunc_wrapper,
        registered_model_name=registered_model_name,
    )
    return model_info
