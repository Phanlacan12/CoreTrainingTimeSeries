from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from mlflow.pyfunc import PythonModel

from tsfb.base.data_module.preprocess_model.base_preprocess import (
    PreprocessData,
    RawDataTuple,
)


class PreprocessPyFunc(PythonModel):
    """
    MLflow pyfunc wrapper that exposes a PreprocessData instance as a model.

    The wrapped PreprocessData object (e.g. DefaultPreprocessData or a custom
    spatio-temporal preprocessor) is serialized together with this PythonModel.

    At inference time, `predict` will:
      - Interpret `model_input` as either a single target DataFrame or a dict
        of DataFrames (target + covariates),
      - Call `self.preprocess.predict(...)`,
      - And return a dictionary of processed series/covariates.
    """

    def __init__(self, preprocess: PreprocessData) -> None:
        """
        Initialize the wrapper with an existing, already-configured preprocessor.

        IMPORTANT:
            Typically you want to log a *fitted* preprocessor (after .fit(...) has
            been called), so that internal scalers/parameters are available for
            `predict(...)`.
        """
        super().__init__()
        self.preprocess = preprocess

    def _to_raw_tuple(  # pylint: disable=unused-argument
        self,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> RawDataTuple:
        """
        Normalize incoming model_input into a RawDataTuple:

            (target_df, past_df, future_df, static_df)

        Supports:
          - model_input as a single pandas.DataFrame (target only)
          - model_input as a dict with keys:
                "target", "past_covariates", "future_covariates",
                "static_covariates"
        """
        if isinstance(model_input, pd.DataFrame):
            # Only target is provided
            return model_input, None, None, None

        if isinstance(model_input, dict):
            target = model_input.get("target")
            past = model_input.get("past_covariates")
            future = model_input.get("future_covariates")
            static = model_input.get("static_covariates")

            if not isinstance(target, pd.DataFrame):
                raise TypeError(
                    "PreprocessPyFunc.predict expects 'target' to be a "
                    "pandas.DataFrame when model_input is a dict."
                )

            past_df = past if isinstance(past, pd.DataFrame) else None
            future_df = future if isinstance(future, pd.DataFrame) else None
            static_df = static if isinstance(static, pd.DataFrame) else None
            return target, past_df, future_df, static_df

        raise TypeError(
            "PreprocessPyFunc.predict only supports model_input as a "
            "pandas.DataFrame or a dict of DataFrames."
        )

    def predict(
        self,
        context: Any,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply the wrapped preprocessor's `predict(...)` on the given raw data.

        Args:
            context: MLflow PythonModelContext (unused here).
            model_input: Either:
                - pandas.DataFrame (target only), or
                - dict with keys:
                    "target", "past_covariates", "future_covariates",
                    "static_covariates"
            params: Optional extra parameters (unused).

        Returns:
            A dictionary with keys:
                - "target_processed"
                - "past_covariates_processed"
                - "future_covariates_processed"
                - "static_processed"
        """
        raw_tuple = self._to_raw_tuple(model_input, params)
        result = self.preprocess.predict(raw_tuple)
        if isinstance(result, dict) and (
            result.get("mode") in {"train", "inference"}
            or "samples" in result
            or "train_samples" in result
        ):
            return result

        target_proc, covariates_proc, static_proc = result

        return {
            "target_processed": target_proc,
            "past_covariates_processed": (
                covariates_proc.get("past_covariates") if covariates_proc else None
            ),
            "future_covariates_processed": (
                covariates_proc.get("future_covariates") if covariates_proc else None
            ),
            "static_processed": static_proc,
        }

    def predict_stream(
        self,
        context: Any,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Optional streaming interface for MLflow pyfunc.

        For now, this simply delegates to `predict`.
        """
        return self.predict(context, model_input, params)
