from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    """Configuration for a single metric."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetricsConfig(BaseModel):
    """
    Configuration for evaluation metrics.
    This corresponds to the 'metrics' list used by the Evaluator.
    """

    metrics: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=lambda: cast(
            List[Union[str, Dict[str, Any]]], ["mae", "mse", "rmse", "mape"]
        ),
        description="List of metric names or dictionaries with metric parameters.",
    )


class StrategyConfig(BaseModel):
    """Base strategy configuration."""

    strategy_name: str
    seed: int = 42
    deterministic: str = "full"
    data_module_config: Dict[str, Any] = Field(default_factory=dict)


class FixedForecastConfig(StrategyConfig):
    """Configuration specific to the FixedForecast strategy."""

    strategy_name: str = "fixed_forecast"
    horizon: int
    save_true_pred: bool = True
    target_channel: Optional[str] = None
