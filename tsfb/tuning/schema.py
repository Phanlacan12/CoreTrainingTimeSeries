# -*- coding: utf-8 -*-
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


class ParameterType(str, Enum):
    """Parameter types."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class TuningParameter(BaseModel):
    """Tuning parameter definition."""

    name: str
    type: ParameterType
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None

    @model_validator(mode="after")
    def validate_param_config(self) -> "TuningParameter":
        """Validate parameter configuration."""
        if self.type in (ParameterType.INT, ParameterType.FLOAT):
            if self.min is None or self.max is None:
                raise ValueError(
                    f"Parameters 'min' and 'max' are required "
                    f"for numeric type {self.type} (param: {self.name})"
                )
            if self.min >= self.max:
                raise ValueError(f"'min' must be less than 'max' for param {self.name}")
        elif self.type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"'choices' must be provided for category type (param: {self.name})"
                )
        return self


class TuningConfig(BaseModel):
    """Tuning configuration schema."""

    parameters: List[TuningParameter]
    n_trials: int = Field(default=20, ge=1)
    metric: str = "rmse"
    metrics_eval: Optional[List[str]] = None
    direction: str = "minimize"
    timeout: Optional[float] = None
    n_jobs: int = 1
    log_top_k_trial: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def normalize_metrics_eval(self) -> "TuningConfig":
        """Ensure metrics_eval always includes the objective metric."""
        metric_names = self.metrics_eval or [self.metric]
        normalized: List[str] = []
        seen: set[str] = set()

        for metric_name in metric_names:
            metric_key = metric_name.lower()
            if metric_key not in seen:
                normalized.append(metric_name)
                seen.add(metric_key)

        objective_key = self.metric.lower()
        if objective_key not in seen:
            normalized.append(self.metric)

        self.metrics_eval = normalized
        return self
