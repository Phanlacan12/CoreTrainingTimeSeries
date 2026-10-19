from __future__ import annotations

# pylint: disable=duplicate-code,ungrouped-imports
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlflow
import torch

from tsfb.base.mlflow.model_logging import (
    build_model_log_metadata,
    build_model_log_tags,
)

try:
    import mlflow.onnx as MLFLOW_ONNX_MODULE  # type: ignore
except Exception:  # pragma: no cover
    MLFLOW_ONNX_MODULE = None  # type: ignore

try:
    import onnx as ONNX_MODULE  # type: ignore
except Exception:  # pragma: no cover
    ONNX_MODULE = None  # type: ignore


_ONNX_EXPORT_METHODS = ("to_onnx", "export_onnx", "save_onnx")


@dataclass(frozen=True)
class ONNXExportInfo:
    """Information about an ONNX export operation.

    Attributes:
        supported: Whether the object supported ONNX export.
        exporter_obj_type: The class name of the exporter object.
        exporter_obj_module: The module of the exporter object.
        export_method: The method name used to export.
        onnx_path: Path to the exported ONNX file.
        meta: Optional metadata recorded alongside the export.
    """

    supported: bool
    exporter_obj_type: str = ""
    exporter_obj_module: str = ""
    export_method: str = ""
    onnx_path: str = ""
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ONNXLogSpec:
    """Configuration for logging an ONNX model to MLflow."""

    artifact_path: str
    registered_model_name: Optional[str] = None
    export_kwargs: Optional[Dict[str, Any]] = None
    log_export_meta: bool = True
    model_role: str = "forecast"


def _find_exporter(obj: Any) -> Optional[Tuple[Any, str]]:
    """Return (exporter_obj, method_name) if the object
    (or obj.model) exposes an ONNX export method.

    Checks for methods listed in _ONNX_EXPORT_METHODS on the object itself first,
    then on the attribute named "model" if present.
    Returns None if no exporter is found.
    """
    for m in _ONNX_EXPORT_METHODS:
        fn = getattr(obj, m, None)
        if callable(fn):
            return obj, m

    inner = getattr(obj, "model", None)
    if inner is not None:
        for m in _ONNX_EXPORT_METHODS:
            fn = getattr(inner, m, None)
            if callable(fn):
                return inner, m

    return None


def supports_onnx_export(model_obj: Any) -> bool:
    """Return True if the provided model object supports ONNX export.

    This is a small helper that wraps _find_exporter for readability.
    """
    return _find_exporter(model_obj) is not None


def export_onnx_to_dir(
    model_obj: Any,
    out_dir: str,
    *,
    filename: str = "model.onnx",
    export_kwargs: Optional[Dict[str, Any]] = None,
) -> ONNXExportInfo:
    """Export a model object to an ONNX file inside `out_dir`.

    The function looks for a supported ONNX export method on `model_obj` (or
    `model_obj.model`) and calls it to write an ONNX file named `filename` into
    `out_dir`.

    Args:
        model_obj: The model or wrapper exposing an ONNX export method.
        out_dir: Directory where the ONNX file will be written.
        filename: Name of the ONNX file to create (default: "model.onnx").
        export_kwargs: Optional keyword arguments forwarded to the export method.

    Returns:
        An ONNXExportInfo instance describing the result. If the object does not
        support ONNX export, returns ONNXExportInfo(supported=False).

    Raises:
        RuntimeError: If the export method ran but the expected ONNX file was
            not created at the expected path.
    """
    export_kwargs = export_kwargs or {}
    found = _find_exporter(model_obj)
    if not found:
        return ONNXExportInfo(supported=False)

    exporter_obj, method_name = found
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, filename)

    getattr(exporter_obj, method_name)(onnx_path, **export_kwargs)

    if not os.path.exists(onnx_path):
        raise RuntimeError(
            f"ONNX export '{method_name}' executed but file not found: {onnx_path}"
        )

    meta = {
        "artifact_kind": "onnx",
        "onnx_file": filename,
        "export_method": method_name,
        "exporter_obj_type": type(exporter_obj).__name__,
        "exporter_obj_module": type(exporter_obj).__module__,
    }

    return ONNXExportInfo(
        supported=True,
        exporter_obj_type=type(exporter_obj).__name__,
        exporter_obj_module=type(exporter_obj).__module__,
        export_method=method_name,
        onnx_path=onnx_path,
        meta=meta,
    )


# ONNX export needs to coordinate model casting, artifact creation, and MLflow
# logging while staying resilient to optional dependencies.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def log_model_onnx_only_if_supported(  # noqa: C901
    model_obj: Any,
    spec: ONNXLogSpec,
) -> Tuple[bool, Optional[Any], Optional[ONNXExportInfo]]:
    """
    ONNX-only logging if supported.
    Return ok=False if:
      - no active run
      - model doesn't support ONNX export
      - mlflow.onnx or onnx is not available
      - export/log fails
    """
    if mlflow.active_run() is None:
        return False, None, None

    if not supports_onnx_export(model_obj):
        return False, None, None

    if ONNX_MODULE is None or MLFLOW_ONNX_MODULE is None:
        return False, None, None

    export_kwargs = {"opset_version": 11, **(spec.export_kwargs or {})}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # FORCE FLOAT32 EXPORT TO AVOID ONNXRuntime
            # NOT_IMPLEMENTED FOR FLOAT64 (DOUBLE)
            # Darts uses float64 by default, which causes AveragePool(11) to fail on CPU
            inner_m = getattr(model_obj, "model", None)
            if inner_m is not None:
                # 1. Cast internal PyTorch module
                pt_model = getattr(inner_m, "model", None)
                if pt_model is not None and hasattr(pt_model, "float"):
                    pt_model.float()
                    # 1b. Manually cast any deep float64 tensors
                    # that .float() missed (e.g. relative_index)
                    # Only needed for TFTModel since it
                    # dynamically generates float64 tensors in __dict__
                    if type(inner_m).__name__ in ["TFTModel"]:

                        def deep_tensor_float_cast(obj, visited=None):
                            if visited is None:
                                visited = set()
                            obj_id = id(obj)
                            if obj_id in visited:
                                return
                            visited.add(obj_id)

                            if hasattr(obj, "__dict__"):
                                for k, v in obj.__dict__.items():
                                    if (
                                        isinstance(v, torch.Tensor)
                                        and v.dtype == torch.float64
                                    ):
                                        setattr(obj, k, v.float())
                                    elif type(v).__name__ in [
                                        "Tensor",
                                        "Parameter",
                                    ] or isinstance(v, torch.nn.Module):
                                        deep_tensor_float_cast(v, visited)
                            elif isinstance(obj, (list, tuple)):
                                for item in obj:
                                    deep_tensor_float_cast(item, visited)

                        deep_tensor_float_cast(pt_model)

                # 2. Cast the Darts TimeSeries used for dummy_input generation
                for attr in [
                    "training_series",
                    "_training_series",
                    "past_covariate_series",
                    "future_covariate_series",
                ]:
                    ts_val = getattr(inner_m, attr, None)
                    if ts_val is not None:
                        if isinstance(ts_val, list):
                            setattr(
                                inner_m, attr, [t.astype("float32") for t in ts_val]
                            )
                        else:
                            setattr(inner_m, attr, ts_val.astype("float32"))

            info = export_onnx_to_dir(
                model_obj,
                tmpdir,
                filename="model.onnx",
                export_kwargs=export_kwargs,
            )
            if not info.supported:
                return False, None, None

            onnx_model = ONNX_MODULE.load(info.onnx_path)

            # Wrap external data into the model to avoid
            # separate .data and .onnx files in MLflow
            try:
                # pylint: disable=import-outside-toplevel
                from onnx.external_data_helper import convert_model_from_external_data

                convert_model_from_external_data(onnx_model)
            except Exception:
                pass

            meta_dict = info.meta if (spec.log_export_meta and info.meta) else None
            extra_tags = (
                {
                    "export_method": (info.export_method or ""),
                    "exporter_module": (info.exporter_obj_module or ""),
                    "exporter_type": (info.exporter_obj_type or ""),
                }
                if spec.log_export_meta
                else None
            )

            model_info = MLFLOW_ONNX_MODULE.log_model(
                onnx_model=onnx_model,
                artifact_path=spec.artifact_path,
                registered_model_name=spec.registered_model_name,
                metadata=build_model_log_metadata(
                    model_role=spec.model_role,
                    kind="onnx",
                    extra_metadata=meta_dict,
                ),
                tags=build_model_log_tags(
                    model_role=spec.model_role,
                    kind="onnx",
                    extra_tags=extra_tags,
                ),
                save_as_external_data=False,
            )

            return True, model_info, info

    except Exception:
        return False, None, None
