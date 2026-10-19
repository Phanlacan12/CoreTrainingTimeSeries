from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from tsfb.base.schema.model_config import ModelConfig


class FineTuneConfig(BaseModel):
    """Configuration for fine-tuning a foundation model."""

    mode: Literal["lora", "full", "prompt"] = "lora"
    epochs: int = 10
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    context_length: Optional[int] = None
    forecast_horizon: int = 96
    stride: int = 1
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    early_stopping_patience: Optional[int] = None
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict)


class FoundationModelConfig(ModelConfig):
    """
    Configuration schema for foundation model adapters.

    Extends ModelConfig with foundation-model-specific fields such as
    pretrained model ID, inference mode, quantization, and fine-tuning config.
    """

    pretrained_model_id: str
    device: str = "cuda"
    inference_mode: Literal["zeroshot", "finetuned", "fewshot"] = "zeroshot"
    finetune_config: Optional[FineTuneConfig] = None
    num_samples: int = 20
    quantization: Optional[Literal["int8", "int4"]] = None
    torch_dtype: str = "bfloat16"
    context_length: Optional[int] = None
    use_covariates: bool = False
    cross_learning: bool = False
