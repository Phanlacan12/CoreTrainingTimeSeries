from __future__ import annotations

from functools import partial
from typing import Dict

from tsfb.base.models.tfb.adapter import TFBModelAdapter
from tsfb.base.models.tfb.registry import MODEL_REGISTRY


def _build_model_info(model_key: str) -> Dict:
    return {
        "model_factory": partial(TFBModelAdapter, model_key),
        "model_name": f"tfb.{model_key}",
    }


for _model_key in MODEL_REGISTRY:
    globals()[_model_key] = _build_model_info(_model_key)


__all__ = list(MODEL_REGISTRY.keys())
