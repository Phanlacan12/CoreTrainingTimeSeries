from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import mlflow
from mlflow.models.model import ModelInfo


def clean_name(value: Any) -> str:
    """Convert an object to a safe MLflow-compatible name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))


def build_registered_model_name(
    *,
    model_name: str,
    register_model: bool,
    registered_model_name: Optional[str] = None,
) -> Optional[str]:
    """Resolve the registered model name to pass into MLflow flavor APIs."""
    if not register_model:
        return None
    return registered_model_name or model_name


def build_model_log_tags(
    *,
    model_role: str,
    kind: str,
    extra_tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build tags for a logged MLflow model."""
    tags: Dict[str, Any] = {
        "model_role": model_role,
        "artifact_kind": kind,
    }
    if extra_tags:
        tags.update(extra_tags)
    return tags


def build_model_log_metadata(
    *,
    model_role: str,
    kind: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build metadata for a logged MLflow model."""
    metadata: Dict[str, Any] = {
        "model_role": model_role,
        "artifact_kind": kind,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


@dataclass(frozen=True)
class LoggedModelResult:
    """Normalized metadata of a logged MLflow model."""

    ok: bool
    kind: str
    model_name: str
    artifact_path: Optional[str] = None
    export_method: Optional[str] = None
    exporter_module: Optional[str] = None
    model_id: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    registered_model_name: Optional[str] = None
    registered_model_version: Optional[int] = None


@dataclass(frozen=True)
class LoggedModelContext:
    """Static context used when mapping ModelInfo to LoggedModelResult."""

    kind: str
    model_name: str
    registered_model_name: Optional[str] = None
    export_method: Optional[str] = None
    exporter_module: Optional[str] = None


@dataclass(frozen=True)
class PyFuncLogSpec:
    """Configuration for logging a pyfunc model via MLflow."""

    model_name: str
    python_model: Any
    register_model: bool
    registered_model_name: Optional[str] = None
    pip_requirements: Optional[Sequence[str]] = None
    model_role: str = "forecast"
    extra_tags: Optional[Dict[str, Any]] = None
    extra_metadata: Optional[Dict[str, Any]] = None


def logged_model_result_from_info(
    *,
    model_info: ModelInfo,
    context: LoggedModelContext,
) -> LoggedModelResult:
    """Map MLflow ModelInfo into the internal logging result contract."""
    return LoggedModelResult(
        ok=True,
        kind=context.kind,
        model_name=context.model_name,
        artifact_path=getattr(model_info, "artifact_path", None),
        export_method=context.export_method,
        exporter_module=context.exporter_module,
        model_id=getattr(model_info, "model_id", None),
        model_uri=getattr(model_info, "model_uri", None),
        run_id=getattr(model_info, "run_id", None),
        registered_model_name=context.registered_model_name,
        registered_model_version=getattr(model_info, "registered_model_version", None),
    )


def log_pyfunc_model(spec: PyFuncLogSpec) -> LoggedModelResult:
    """Log a PythonModel via MLflow's native pyfunc flavor API."""
    effective_reg_name = build_registered_model_name(
        model_name=spec.model_name,
        register_model=spec.register_model,
        registered_model_name=spec.registered_model_name,
    )
    model_info = mlflow.pyfunc.log_model(
        artifact_path=spec.model_name,
        python_model=spec.python_model,
        registered_model_name=effective_reg_name,
        pip_requirements=(
            list(spec.pip_requirements) if spec.pip_requirements else None
        ),
        metadata=build_model_log_metadata(
            model_role=spec.model_role,
            kind="pyfunc",
            extra_metadata=spec.extra_metadata,
        ),
        tags=build_model_log_tags(
            model_role=spec.model_role,
            kind="pyfunc",
            extra_tags=spec.extra_tags,
        ),
    )
    return logged_model_result_from_info(
        model_info=model_info,
        context=LoggedModelContext(
            kind="pyfunc",
            model_name=spec.model_name,
            registered_model_name=effective_reg_name,
        ),
    )
