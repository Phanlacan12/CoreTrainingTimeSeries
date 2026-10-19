from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import pandas as pd

from tsfb.base.data_module.preprocess_model.base_preprocess import PreprocessData
from tsfb.base.data_module.preprocess_model.components.dataset_factory import (
    build_loader,
)
from tsfb.base.data_module.preprocess_model.components.window_builder import (
    build_inference_samples,
    build_training_samples,
)
from tsfb.base.schema.tfb_family import ModelFamily


class DeepWindowPreprocess(PreprocessData, ABC):
    """Common deep-learning preprocessing with dataset/dataloader logic embedded."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.family = ModelFamily(config["family"])
        self.seq_len = int(config["seq_len"])
        self.pred_len = int(config["pred_len"])
        self.label_len = int(config.get("label_len", max(1, self.seq_len // 2)))
        self.freq = config.get("freq", "D")
        self.batch_size = int(config.get("batch_size", 32))
        self.num_workers = int(config.get("num_workers", 0))
        self.target_columns = list(config.get("target_columns", []))
        self._training_payload: Optional[Dict[str, Any]] = None
        self._inference_payload: Optional[Dict[str, Any]] = None

    def _value_columns(self, frame: pd.DataFrame) -> list[str]:
        cols = self.target_columns or list(frame.columns)
        return [col for col in cols if col in frame.columns]

    def fit(
        self,
        data: Any,
        *,
        train_ratio_in_tv: float = 1.0,
        covariates: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fit preprocessing state and cache train/validation window payloads."""
        del covariates, kwargs
        frame = self._coerce_frame(data)
        value_cols = self._value_columns(frame)
        train_samples, val_samples = build_training_samples(
            frame,
            family=self.family,
            value_cols=value_cols,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            freq=self.freq,
            train_ratio=train_ratio_in_tv,
        )
        self._training_payload = {
            "mode": "train",
            "family": self.family.value,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "meta": {"value_cols": value_cols},
        }
        return self._training_payload

    def predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build and cache inference window payloads for a prediction frame."""
        del kwargs
        frame = self._coerce_frame(data)
        value_cols = self._value_columns(frame)
        inference_samples = build_inference_samples(
            frame,
            family=self.family,
            value_cols=value_cols,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            freq=self.freq,
        )
        self._inference_payload = {
            "mode": "inference",
            "family": self.family.value,
            "samples": inference_samples,
            "meta": {"value_cols": value_cols},
        }
        return self._inference_payload

    def inverse_predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Return predictions unchanged for models without inverse transforms."""
        del kwargs
        return data

    def build_train_loader(self):
        """Build the training DataLoader from the cached fit payload."""
        if not self._training_payload:
            raise RuntimeError("Call fit() before build_train_loader().")
        return build_loader(
            self._training_payload["train_samples"],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def build_val_loader(self):
        """Build the validation DataLoader when validation samples exist."""
        if not self._training_payload:
            raise RuntimeError("Call fit() before build_val_loader().")
        samples = self._training_payload["val_samples"]
        if not samples or len(next(iter(samples.values()))) == 0:
            return None
        return build_loader(
            samples,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def build_inference_loader(self):
        """Build the inference DataLoader from the cached predict payload."""
        if not self._inference_payload:
            raise RuntimeError("Call predict() before build_inference_loader().")
        return build_loader(
            self._inference_payload["samples"],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def get_training_artifacts(self) -> Dict[str, Any]:
        """Return the cached training payload assembled during `fit()`."""
        if not self._training_payload:
            raise RuntimeError("Call fit() before get_training_artifacts().")
        return self._training_payload

    def get_inference_artifacts(self) -> Dict[str, Any]:
        """Return the cached inference payload assembled during `predict()`."""
        if not self._inference_payload:
            raise RuntimeError("Call predict() before get_inference_artifacts().")
        return self._inference_payload

    @staticmethod
    def _coerce_frame(data: Any) -> pd.DataFrame:
        """Normalize supported input payloads to a standalone DataFrame copy."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, tuple) and data and isinstance(data[0], pd.DataFrame):
            return data[0].copy()
        raise TypeError(
            "TFB preprocess expects a pandas.DataFrame or a tuple whose first element "
            "is a pandas.DataFrame."
        )


class TFBPreprocess(DeepWindowPreprocess):
    """Ungrouped deep-learning preprocess used by tfb models."""
