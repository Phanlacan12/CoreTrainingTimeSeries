from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tsfb.base.data_module.preprocess_model.tfb_preprocess import DeepWindowPreprocess


class TFBGroupedPreprocess(DeepWindowPreprocess):
    """Grouped deep-learning preprocess with per-entity window generation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.group_col = config["group_col"]
        self.time_col = config["time_col"]

    def fit(
        self,
        data: Any,
        *,
        train_ratio_in_tv: float = 1.0,
        covariates: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        frame = self._coerce_grouped_frame(data)
        value_cols = self._value_columns(frame)
        train_parts: List[Dict[str, Any]] = []
        val_parts: List[Dict[str, Any]] = []

        for _, group_df in frame.groupby(self.group_col, sort=True):
            group_df = (
                group_df.sort_values(self.time_col)
                .set_index(self.time_col, drop=True)[value_cols]
                .copy()
            )
            payload = super().fit(group_df, train_ratio_in_tv=train_ratio_in_tv)
            train_parts.append(payload["train_samples"])
            val_parts.append(payload["val_samples"])

        self._training_payload = {
            "mode": "train",
            "family": self.family.value,
            "train_samples": self._concat_sample_parts(train_parts),
            "val_samples": self._concat_sample_parts(val_parts),
            "meta": {"value_cols": value_cols},
        }
        return self._training_payload

    def predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        frame = self._coerce_grouped_frame(data)
        value_cols = self._value_columns(frame)
        sample_parts: List[Dict[str, Any]] = []
        group_ids: List[Any] = []
        prediction_index: List[Any] = []

        for gid, group_df in frame.groupby(self.group_col, sort=True):
            group_df = (
                group_df.sort_values(self.time_col)
                .set_index(self.time_col, drop=True)[value_cols]
                .copy()
            )
            payload = super().predict(group_df)
            sample_parts.append(payload["samples"])
            group_ids.extend([gid] * len(payload["samples"]["prediction_index"]))
            prediction_index.extend(payload["samples"]["prediction_index"])

        samples = self._concat_sample_parts(sample_parts)
        samples["prediction_index"] = prediction_index
        samples["group_id"] = group_ids
        self._inference_payload = {
            "mode": "inference",
            "family": self.family.value,
            "samples": samples,
            "meta": {
                "value_cols": value_cols,
                "group_col": self.group_col,
                "time_col": self.time_col,
            },
        }
        return self._inference_payload

    def _coerce_grouped_frame(self, data: Any) -> pd.DataFrame:
        frame = self._coerce_frame(data)
        if self.group_col not in frame.columns or self.time_col not in frame.columns:
            raise ValueError(
                f"Grouped preprocess needs columns "
                f"{self.group_col!r} and {self.time_col!r}."
            )
        result = frame.copy()
        result[self.time_col] = pd.to_datetime(result[self.time_col])
        return result

    @staticmethod
    def _concat_sample_parts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if not parts:
            return merged
        keys = [key for key in parts[0] if key not in {"prediction_index", "group_id"}]
        for key in keys:
            arrays = [part[key] for part in parts if key in part and len(part[key]) > 0]
            if arrays:
                merged[key] = np.concatenate(arrays, axis=0)
            else:
                merged[key] = parts[0][key]
        return merged
