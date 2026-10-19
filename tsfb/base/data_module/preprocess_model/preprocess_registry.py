from typing import Dict, Type

from pydantic import BaseModel

from tsfb.base.data_module.preprocess_model import (
    ASTGCRNPreprocess,
    DefaultPreprocessData,
    StaticGroupedPreprocess,
)
from tsfb.base.data_module.preprocess_model.base_preprocess import PreprocessData

PREPROCESS_REGISTRY: Dict[str, Type[PreprocessData]] = {
    "default": DefaultPreprocessData,
    # "spatio": SpatioPreprocess,
    "static_grouped": StaticGroupedPreprocess,
    "astgcrn": ASTGCRNPreprocess,
}


class PreprocessConfig(BaseModel):
    """Configuration for a single preprocessing model."""

    type: str = "default"
    params: dict = {}


class TrainingDataConfig(BaseModel):
    """
    The system records configuration information for constructing
    training data from different perspectives in order to evaluate
    model results for analyzing and forecasting the demand for
    online public service usage based on residency status.
    """

    training_data_aspect_config: PreprocessConfig


def build_preprocess_from_config(config: dict) -> PreprocessData:
    """Builds a preprocess model instance from the given config."""
    # Validate and parse config using Pydantic
    parsed_config = TrainingDataConfig(**config)
    preprocess_cfg = parsed_config.training_data_aspect_config

    preprocess_type = preprocess_cfg.type
    preprocess_params = preprocess_cfg.params

    if preprocess_type not in PREPROCESS_REGISTRY:
        raise ValueError(f"Unknown preprocess type: {preprocess_type}")

    preprocess_class: Type[PreprocessData] = PREPROCESS_REGISTRY[preprocess_type]

    return preprocess_class(preprocess_params)
