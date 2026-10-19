import abc
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from tsfb.base.models.base_model import ModelBase
from tsfb.base.schema.foundation_config import FineTuneConfig


class FoundationModelBase(ModelBase, metaclass=abc.ABCMeta):
    """
    Extended base class for foundation model adapters.

    Adds zero-shot, fine-tuning, few-shot, and probabilistic forecasting
    capabilities on top of the standard ModelBase interface.

    Lifecycle:
      - Zero-shot:  forecast_fit() is a no-op; call forecast() directly.
      - Fine-tuned: forecast_fit() delegates to finetune(); then forecast().
      - Few-shot:   call set_few_shot_context() before forecast().
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        context_length: Optional[int] = 512,
        torch_dtype: str = "bfloat16",
        finetune_config: Optional[dict] = None,
        **_kwargs,
    ) -> None:
        self._model_id = pretrained_model_id
        self._device = device
        self._inference_mode = inference_mode
        self._context_length = context_length
        self._torch_dtype_str = torch_dtype
        self._raw_finetune_config = finetune_config
        self._model: Any = None

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "FoundationModelBase":
        if self._inference_mode == "finetuned":
            ft_cfg = FineTuneConfig(**(self._raw_finetune_config or {}))
            self.finetune(train_data, ft_cfg)
        return self

    @abc.abstractmethod
    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """Run zero-shot or post-finetuned inference."""

    def finetune(
        self,
        train_data: pd.DataFrame,
        config: "FineTuneConfig",
        val_data: Optional[pd.DataFrame] = None,
    ) -> "FoundationModelBase":
        """
        Fine-tune the foundation model on domain-specific data.

        Override in subclasses that support fine-tuning.
        """
        raise NotImplementedError(f"{self.model_name} does not support fine-tuning.")

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Return a sample matrix of shape (num_samples, horizon).

        Override in subclasses that support probabilistic output.
        The default falls back to deterministic forecast replicated across samples.
        """
        raise NotImplementedError(
            f"{self.model_name} does not support probabilistic forecasting."
        )

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> "Any":
        """
        Return a DistributionForecast fitted to this model's probabilistic output.

        Default: fit a parametric distribution to samples from forecast_probabilistic()
        using moment-matching (from_samples). Adapters with native distribution output
        should override this for exact parameters.

        Args:
            horizon: Forecast horizon.
            series: Input time series.
            distribution_name: 'normal' | 'student_t' | 'negative_binomial'.
            num_samples: Samples drawn for moment-matching (ignored if overridden).

        Returns:
            DistributionForecast object.
        """
        # pylint: disable=import-outside-toplevel
        from tsfb.base.forecasting.distribution_forecast import DistributionForecast

        samples = self.forecast_probabilistic(
            horizon, series, num_samples=num_samples, covariates=covariates
        )
        return DistributionForecast.from_samples(samples, distribution_name)

    def set_few_shot_context(self, context: pd.DataFrame) -> "FoundationModelBase":
        """
        Inject few-shot context series before calling forecast().

        Some foundation models use additional context windows as 'prompts'.
        Override in subclasses that support this pattern.
        """
        raise NotImplementedError(
            f"{self.model_name} does not support few-shot context injection."
        )

    # ------------------------------------------------------------------
    # Capability flags — subclasses override as needed
    # ------------------------------------------------------------------

    @property
    def supports_zeroshot(self) -> bool:
        """Whether the adapter supports zero-shot forecasting."""
        return True

    @property
    def supports_finetuning(self) -> bool:
        """Whether the adapter supports model fine-tuning."""
        return False

    @property
    def supports_probabilistic(self) -> bool:
        """Whether the adapter supports probabilistic forecasting."""
        return False

    @property
    def supports_distribution(self) -> bool:
        """Whether the adapter natively exposes distribution parameters.

        When False, forecast_distribution() falls back to fitting a parametric
        distribution to samples from forecast_probabilistic() via moment-matching.
        Adapters that return exact distribution params should override to True.
        """
        return self.supports_probabilistic

    @property
    def output_type(self) -> str:
        """Preferred output type: 'samples' | 'distribution'.

        Strategy layer uses this to decide which forecast path to call.
        Override in adapters that natively produce distribution parameters.
        """
        return "samples"

    @property
    def supports_fewshot(self) -> bool:
        """Whether the adapter supports few-shot context injection."""
        return False

    # ------------------------------------------------------------------
    # Batch forecasting — shared loop; subclasses implement _infer_contexts
    # ------------------------------------------------------------------

    def batch_forecast(self, horizon: int, batch_maker: Any, **_kwargs) -> np.ndarray:
        """Batch forecast loop for foundation models (standard BatchMaker protocol).

        Subclasses that support batching must set supports_batch_forecast = True
        and implement _infer_contexts(). Returns shape (n_windows, horizon).
        """
        batch_size = getattr(self, "_batch_size", 32)
        win_size = self._compute_win_size(batch_maker)
        all_preds: list = []
        while batch_maker.has_more_batches():
            batch = batch_maker.make_batch(batch_size=batch_size, win_size=win_size)
            raw = batch["input"]  # (B, win_size[, n_features])
            contexts = [raw[i].astype(np.float32) for i in range(len(raw))]
            all_preds.append(self._infer_contexts(contexts, horizon))
        return np.concatenate(all_preds, axis=0)  # (n_windows, horizon)

    def _compute_win_size(self, batch_maker: Any) -> int:
        """Cap context window to the minimum available lookback across windows."""
        inner = getattr(batch_maker, "_batch_maker", None)
        win_size = self._context_length or 512
        if inner is not None and len(getattr(inner, "index_list", [])) > 0:
            win_size = min(win_size, *inner.index_list)
        return win_size

    def _infer_contexts(self, contexts: list, horizon: int) -> np.ndarray:
        """Run inference on a list of 1-D context arrays; return (B, horizon).

        Override in subclasses that set supports_batch_forecast = True.
        """
        raise NotImplementedError(
            f"{self.model_name} does not implement _infer_contexts()."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_torch_dtype(self) -> torch.dtype:
        dtype_map: dict = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        return dtype_map.get(self._torch_dtype_str, torch.bfloat16)

    def _series_to_tensor(self, series: pd.DataFrame) -> "np.ndarray":
        """Convert a single-column or multi-column DataFrame to a float32 array."""
        return series.select_dtypes(include="number").values.astype("float32")
