# -*- coding: utf-8 -*-
"""
Custom MLflow PyFunc wrapper for models inheriting from ModelBase.
Allows MLflow to save, load, train, and predict with your custom time series models.
"""

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from tsfb.base.models.base_model import ModelBase


class ModelBasePyFunc(PythonModel):
    """
    Wrapper class so that MLflow can log, load, and use your ModelBase models.

    - During training: call .fit(train_df)
    - During inference: call .predict(context, test_df)
    - When MLflow loads the model: automatically unpickles the original model
    """

    MODEL_FILE_NAME = "model.pkl"

    def __init__(self, model_obj: ModelBase, horizon: int = 1):
        """
        Args:
            model_obj (ModelBase): An instance of a model inheriting from ModelBase.
            horizon (int): Forecasting horizon (number of steps ahead).
        """
        self.model: ModelBase = model_obj
        self.horizon = horizon
        self.is_fitted = False

    def fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ):
        """
        Train the model using time series data.

        Args:
            train_data (pd.DataFrame): Training data.
            covariates (dict, optional): Optional covariate data.
            train_ratio_in_tv (float): Ratio of train vs validation split.
        """
        if not isinstance(self.model, ModelBase):
            raise TypeError("Underlying model must inherit from ModelBase")

        self.model.forecast_fit(
            train_data=train_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            **kwargs,
        )
        self.is_fitted = True
        return self

    def predict(
        self, context: Any, model_input: Any, params: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Forecast using the trained model. Called by
        MLflow when you use model.predict(data).

        Args:
            context: MLflow runtime context (ignored here).
            model_input (pd.DataFrame): Input time series data.
            params (dict, optional): Additional parameters (ignored).

        Returns:
            pd.DataFrame: Forecasted values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

        arr = self.model.forecast(
            horizon=self.horizon,
            series=model_input,
            covariates=None,
        )

        if isinstance(arr, np.ndarray):
            return pd.DataFrame(arr, columns=model_input.columns)
        if isinstance(arr, pd.DataFrame):
            return arr
        return pd.DataFrame(np.array(arr))

    def predict_stream(
        self, context: Any, model_input: Any, params: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Optional MLflow PyFunc streaming prediction method.
        For now, just call the standard predict method.
        """
        return NotImplementedError

    def save_context(self, path: str):
        """
        Called by MLflow when saving the model.
        Pickles the underlying ModelBase instance.
        """
        model_path = os.path.join(path, self.MODEL_FILE_NAME)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_context(self, context):
        """
        Called by MLflow when loading the model. Unpickles the saved ModelBase instance.
        """
        model_path = os.path.join(context.artifacts["model_dir"], self.MODEL_FILE_NAME)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
