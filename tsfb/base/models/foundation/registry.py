from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FoundationModelSpec:
    """Metadata about a registered foundation model."""

    adapter_class: str
    default_model_id: str
    supports_finetuning: bool = False
    supports_probabilistic: bool = False
    supports_fewshot: bool = False
    install_hint: Optional[str] = None
    extra_defaults: Dict = field(default_factory=dict)


FOUNDATION_MODEL_REGISTRY: Dict[str, FoundationModelSpec] = {
    "chronos": FoundationModelSpec(
        adapter_class=(
            "tsfb.base.models.foundation.adapters.chronos_adapter.ChronosAdapter"
        ),
        default_model_id="autogluon/chronos-2-small",
        supports_finetuning=True,
        supports_probabilistic=True,
        supports_fewshot=False,
        install_hint="pip install chronos-forecasting",
    ),
    "timesfm": FoundationModelSpec(
        adapter_class=(
            "tsfb.base.models.foundation.adapters.timesfm_adapter.TimesFMAdapter"
        ),
        default_model_id="google/timesfm-2.5-200m-pytorch",
        supports_finetuning=False,
        supports_probabilistic=True,
        supports_fewshot=False,
        install_hint="pip install 'timesfm>=2.0.0'",
    ),
    "tirex": FoundationModelSpec(
        adapter_class=(
            "tsfb.base.models.foundation.adapters.tirex_adapter.TiRexAdapter"
        ),
        default_model_id="NX-AI/TiRex-2",
        supports_finetuning=False,
        supports_probabilistic=True,
        supports_fewshot=False,
        install_hint=(
            "python -m pip install --no-deps tirex-2==0.1.1 xlstm==2.0.5 "
            "flashrnn==1.0.6 mlstm-kernels==2.0.4"
        ),
    ),
    "moirai": FoundationModelSpec(
        adapter_class=(
            "tsfb.base.models.foundation.adapters.moirai_adapter.MoiraiAdapter"
        ),
        default_model_id="Salesforce/moirai-1.0-R-small",
        supports_finetuning=True,
        supports_probabilistic=True,
        supports_fewshot=False,
        install_hint="pip install uni2ts",
    ),
    "moment": FoundationModelSpec(
        adapter_class=(
            "tsfb.base.models.foundation.adapters.moment_adapter.MomentAdapter"
        ),
        default_model_id="AutonLab/MOMENT-1-large",
        supports_finetuning=True,
        supports_probabilistic=False,
        supports_fewshot=False,
        install_hint="pip install momentfm",
    ),
    "kronos": FoundationModelSpec(
        adapter_class=(
            "tsfb.base.models.foundation.adapters.kronos_adapter.KronosAdapter"
        ),
        default_model_id="shiyu-coder/Kronos-Base",
        supports_finetuning=True,
        supports_probabilistic=False,
        supports_fewshot=False,
        install_hint="pip install git+https://github.com/shiyu-coder/Kronos.git",
    ),
}


def get_foundation_spec(name: str) -> FoundationModelSpec:
    """Return the FoundationModelSpec for a registered model key."""
    key = name.lower()
    if key not in FOUNDATION_MODEL_REGISTRY:
        available = ", ".join(FOUNDATION_MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown foundation model '{name}'. Available: {available}")
    return FOUNDATION_MODEL_REGISTRY[key]
