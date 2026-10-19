from typing import Any, Dict

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """
    Configuration schema for a single forecasting model.

    Attributes:
        model_name: The name of the Darts model class (e.g., 'LinearRegressionModel').
        model_args: Hyperparameters to pass to the model constructor.
        allow_fit_on_eval: If True, allows the model to be
            re-fitted during evaluation/inference.
        supports_validation: If True, indicates the model supports a
            validation set during training.
    """

    model_name: str
    model_args: Dict[str, Any] = Field(default_factory=dict)
    allow_fit_on_eval: bool = False
    supports_validation: bool = False
