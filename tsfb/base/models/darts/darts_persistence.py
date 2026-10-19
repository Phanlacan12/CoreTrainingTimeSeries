from __future__ import annotations

import importlib
import json
import os
from typing import TYPE_CHECKING

import cloudpickle
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils.onnx_utils import prepare_onnx_inputs

from tsfb.base.schema.darts_config import DartsConfig

if TYPE_CHECKING:
    from tsfb.base.models.darts.darts_models import DartsModelAdapter


def save_darts_adapter(adapter: "DartsModelAdapter", path: str) -> None:
    """Save a DartsModelAdapter and its Darts model to path"""
    if adapter.model is None:
        raise ValueError("Cannot save DartsModelAdapter before fitting the model.")

    os.makedirs(path, exist_ok=True)

    metadata = {
        "adapter_module": type(adapter).__module__,
        "adapter_class": type(adapter).__name__,
        "darts_model_module": type(adapter.model).__module__,
        "darts_model_class": type(adapter.model).__name__,
        "model_name": adapter._model_name,  # pylint: disable=protected-access
        "allow_fit_on_eval": adapter.allow_fit_on_eval,
        "supports_validation": adapter.supports_validation,
        "config": adapter.config.to_dict(),
        "_in_columns": adapter._in_columns,  # pylint: disable=protected-access
        "_cov_columns": adapter._cov_columns,  # pylint: disable=protected-access
        "train_ratio_in_tv": adapter.train_ratio_in_tv,
    }

    if isinstance(adapter.model, TorchForecastingModel):
        pt_file = os.path.join(path, "darts_model.pt")
        adapter.model.save(pt_file)

        onnx_file = os.path.join(path, "darts_model.onnx")
        # Use adapter exporter so it matches onnx_handle behavior.
        adapter.to_onnx(onnx_file, export_params=True)

        metadata.update(
            {
                "artifact_kind": "onnx",
                "torch_file": "darts_model.pt",  # keep for preparing inputs / fallback
                "onnx_file": "darts_model.onnx",
                "preferred_runtime": "onnx",
            }
        )
    else:
        pkl_file = os.path.join(path, "darts_model.pkl")
        adapter.model.save(pkl_file)
        metadata.update(
            {
                "artifact_kind": "pickle",
                "pkl_file": "darts_model.pkl",
                "preferred_runtime": "darts",
            }
        )

    meta_file = os.path.join(path, "metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # pylint: disable=protected-access
    extra_state = {
        "_full_future_cov": adapter._full_future_cov,
        "_static_transformer": adapter._static_transformer,
        "onnx_path": adapter.onnx_path,
        "runtime": getattr(adapter, "runtime", "darts"),
    }
    state_file = os.path.join(path, "adapter_state.pkl")
    with open(state_file, "wb") as f:
        cloudpickle.dump(extra_state, f)


# pylint: disable=too-many-locals
# pylint: disable=protected-access
def load_darts_adapter(
    cls: type["DartsModelAdapter"], path: str
) -> "DartsModelAdapter":
    """Load a DartsModelAdapter instance from path"""
    meta_file = os.path.join(path, "metadata.json")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"metadata.json not found in {path}")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    darts_module_name = metadata["darts_model_module"]
    darts_class_name = metadata["darts_model_class"]
    darts_module = importlib.import_module(darts_module_name)
    darts_model_cls = getattr(darts_module, darts_class_name)

    if not issubclass(darts_model_cls, ForecastingModel):
        raise TypeError(
            f"Loaded class {darts_class_name} from {darts_module_name} "
            "is not a Darts ForecastingModel."
        )

    adapter = cls(
        model_class=darts_model_cls,
        model_args={},  # placeholder
        model_name=metadata.get("model_name"),
        allow_fit_on_eval=metadata.get("allow_fit_on_eval", False),
        supports_validation=metadata.get("supports_validation", False),
    )
    adapter.runtime = metadata.get("preferred_runtime", "darts")

    adapter.config = DartsConfig(**metadata["config"])
    adapter._in_columns = metadata.get("_in_columns")
    adapter._cov_columns = metadata.get(
        "_cov_columns"
    )  # pylint: disable=protected-access
    adapter.train_ratio_in_tv = metadata.get("train_ratio_in_tv", 1.0)

    artifact_kind = metadata.get("artifact_kind", "pickle")

    if artifact_kind in ("onnx", "torch"):
        pt_path = os.path.join(path, metadata.get("torch_file", "darts_model.pt"))
        adapter.model = (
            darts_model_cls.load(pt_path) if os.path.exists(pt_path) else None
        )

        onnx_path = os.path.join(path, metadata.get("onnx_file", "darts_model.onnx"))
        adapter.onnx_path = onnx_path if os.path.exists(onnx_path) else None

    else:
        pkl_path = os.path.join(path, metadata.get("pkl_file", "darts_model.pkl"))
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Darts model file not found: {pkl_path}")
        adapter.model = darts_model_cls.load(pkl_path)

    # restore extra state
    state_file = os.path.join(path, "adapter_state.pkl")
    if os.path.exists(state_file):
        with open(state_file, "rb") as f:
            extra_state = cloudpickle.load(f)
        adapter._full_future_cov = extra_state.get(
            "_full_future_cov"
        )  # pylint: disable=protected-access
        adapter._static_transformer = extra_state.get(
            "_static_transformer"
        )  # pylint: disable=protected-access
        adapter.onnx_path = extra_state.get("onnx_path") or adapter.onnx_path
        adapter.runtime = extra_state.get("runtime") or adapter.runtime

    return adapter


def _import_prepare_onnx_inputs():
    # Kept for backward compatibility with callers expecting this function.
    return prepare_onnx_inputs
