from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlflow

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.mlflow.model_logging import (
    LoggedModelContext,
    LoggedModelResult,
    PyFuncLogSpec,
    build_registered_model_name,
    clean_name,
    log_pyfunc_model,
    logged_model_result_from_info,
)
from tsfb.base.mlflow.onnx_handle import ONNXLogSpec, log_model_onnx_only_if_supported
from tsfb.base.mlflow.pyfunc_model import ModelBasePyFunc

logger = logging.getLogger(__name__)


def get_horizon_from_cfg(cfg: Dict[str, Any]) -> int:
    """Get forecasting horizon from evaluation config."""
    evaluation = cfg.get("evaluation")
    if evaluation is None:
        raise KeyError("'evaluation' section missing in config.")

    strategy_args = evaluation.get("strategy_args")
    if strategy_args is None:
        raise KeyError("'strategy_args' missing in evaluation.")

    return int(strategy_args.get("horizon", 1))


def iter_leaf_models(obj: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield (name_suffix, model_obj) for all leaf models."""
    if obj is None:
        return

    if isinstance(obj, tuple) and len(obj) > 0:
        yield from iter_leaf_models(obj[0], prefix=prefix)
        return

    if hasattr(obj, "model") and getattr(obj, "model") is not None:
        yield (prefix, getattr(obj, "model"))
        return

    if hasattr(obj, "models") and isinstance(getattr(obj, "models"), dict):
        for k, v in getattr(obj, "models").items():
            key = clean_name(k)
            new_prefix = f"{prefix}_{key}" if prefix else key
            yield from iter_leaf_models(v, prefix=new_prefix)
        return

    yield (prefix or clean_name(type(obj).__name__), obj)


def parse_group_cols(sub_key: Any) -> List[str]:
    """Parse grouped column names from a composite key."""
    if isinstance(sub_key, str) and "," in sub_key:
        return [clean_name(x.strip()) for x in sub_key.split(",") if x.strip()]
    return []


def build_model_name(
    *,
    approach_name: str,
    sub_key: Any,
    suffix: str,
    group_cols: List[str],
) -> str:
    """Build a unique MLflow model name."""
    base_name = f"{clean_name(approach_name)}_{clean_name(sub_key)}"
    suffix_clean = clean_name(suffix) if suffix else ""

    if group_cols and suffix_clean in group_cols:
        return f"{clean_name(approach_name)}_{suffix_clean}"

    return f"{base_name}_{suffix_clean}" if suffix_clean else base_name


def log_one_model_best_effort(
    *,
    model_obj: Any,
    model_name: str,
    horizon: int,
    prefer_onnx: bool = True,
    registered_model_name: Optional[str] = None,
    register_model: bool = True,
    # pylint: disable=too-many-arguments,too-many-locals
) -> LoggedModelResult:
    """Log a model via ONNX if possible, else fallback to PyFunc.

    Args:
        model_obj: The model object to log.
        model_name: Name used for the artifact path.
        horizon: Forecasting horizon for PyFunc wrapper.
        prefer_onnx: If True, try ONNX export first.
        registered_model_name: If provided, register the model under this name.
            If None, defaults to model_name for registration.
    """
    effective_reg_name = build_registered_model_name(
        model_name=model_name,
        register_model=register_model,
        registered_model_name=registered_model_name,
    )

    if prefer_onnx:
        ok, model_info, info = log_model_onnx_only_if_supported(
            model_obj=model_obj,
            spec=ONNXLogSpec(
                artifact_path=model_name,
                registered_model_name=effective_reg_name,
                export_kwargs={},
                log_export_meta=True,
                model_role="forecast",
            ),
        )
        if ok and model_info is not None:
            return logged_model_result_from_info(
                model_info=model_info,
                context=LoggedModelContext(
                    kind="onnx",
                    model_name=model_name,
                    registered_model_name=effective_reg_name,
                    export_method=getattr(info, "export_method", None),
                    exporter_module=getattr(info, "exporter_obj_module", None),
                ),
            )

    try:
        return log_pyfunc_model(
            PyFuncLogSpec(
                model_name=model_name,
                python_model=ModelBasePyFunc(model_obj, horizon=horizon),
                register_model=register_model,
                registered_model_name=registered_model_name,
                pip_requirements=["cloudpickle", "darts"],
                model_role="forecast",
                extra_metadata={"python_model": "ModelBasePyFunc"},
            )
        )
    except Exception as e:
        logger.warning("Failed to log MLflow forecast model '%s': %s", model_name, e)
        return LoggedModelResult(
            ok=False,
            kind="pyfunc",
            model_name=model_name,
        )


def log_single_approach_model(
    *,
    approach: ForecastingApproach,
    approach_name: str,
    horizon: int,
    experiment_name: str,
    register_model: bool = True,
) -> Optional[LoggedModelResult]:
    """Log a single model from an approach if present."""
    if not (hasattr(approach, "model") and getattr(approach, "model") is not None):
        return None

    model_obj = getattr(approach, "model")
    model_name = clean_name(approach_name)
    model_name = f"{clean_name(experiment_name)}_{model_name}"

    return log_one_model_best_effort(
        model_obj=model_obj,
        model_name=model_name,
        horizon=horizon,
        prefer_onnx=True,
        register_model=register_model,
    )


def log_multi_models_under_approach_models(
    *,
    approach: ForecastingApproach,
    approach_name: str,
    horizon: int,
    experiment_name: str,
    register_model: bool = True,
) -> List[LoggedModelResult]:
    """Log all leaf models under approach.models."""
    results: List[LoggedModelResult] = []

    if not (
        hasattr(approach, "models") and isinstance(getattr(approach, "models"), dict)
    ):
        return results

    for sub_key, sub_val in getattr(approach, "models").items():
        leafs = list(iter_leaf_models(sub_val))
        if not leafs:
            continue

        group_cols = parse_group_cols(sub_key)

        for suffix, model_obj in leafs:
            model_name = build_model_name(
                approach_name=approach_name,
                sub_key=sub_key,
                suffix=suffix,
                group_cols=group_cols,
            )
            model_name = f"{clean_name(experiment_name)}_{model_name}"

            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.set_tag("approach_name", approach_name)
                mlflow.set_tag("sub_model_name", str(sub_key))
                if suffix:
                    mlflow.set_tag("leaf_model", suffix)

                res = log_one_model_best_effort(
                    model_obj=model_obj,
                    model_name=model_name,
                    horizon=horizon,
                    prefer_onnx=True,
                    register_model=register_model,
                )
                results.append(res)

    return results


def log_approach_models(
    *,
    cfg: Dict[str, Any],
    approach: ForecastingApproach,
    approach_name: str,
    mlflow_enabled: bool,
    run_logger: Any,
) -> None:
    """Log all models of an approach to MLflow."""
    if type(approach).__name__ == "OldONNXApproach":
        run_logger.info(
            "Skipping model logging for OldONNXApproach to avoid re-registration"
        )
        return

    if not mlflow_enabled:
        run_logger.info(
            "MLflow disabled – skipping model logging for %s", approach_name
        )
        return

    horizon = get_horizon_from_cfg(cfg)
    experiment_name = cfg.get("mlflow", {}).get(
        "experiment_name", "ForecastingBenchmark"
    )
    register_model = cfg.get("mlflow", {}).get("register_model", True)

    run_logger.info(
        "Using MLflow artifact base: %s", Path(mlflow.get_artifact_uri()).resolve()
    )

    try:
        res = log_single_approach_model(
            approach=approach,
            approach_name=approach_name,
            horizon=horizon,
            experiment_name=experiment_name,
            register_model=register_model,
        )
        if res is not None:
            run_logger.info(
                "Logged forecast model '%s' (kind=%s, artifact_path=%s, uri=%s, "
                "run_id=%s, registry=%s, version=%s, export_method=%s, module=%s)",
                res.model_name,
                res.kind,
                res.artifact_path,
                res.model_uri,
                res.run_id,
                res.registered_model_name,
                res.registered_model_version,
                res.export_method,
                res.exporter_module,
            )
            return

        results = log_multi_models_under_approach_models(
            approach=approach,
            approach_name=approach_name,
            horizon=horizon,
            experiment_name=experiment_name,
            register_model=register_model,
        )
        if results:
            for r in results:
                run_logger.info(
                    "Logged forecast sub-model '%s' (kind=%s, artifact_path=%s, "
                    "uri=%s, run_id=%s, registry=%s, version=%s, export_method=%s, "
                    "module=%s)",
                    r.model_name,
                    r.kind,
                    r.artifact_path,
                    r.model_uri,
                    r.run_id,
                    r.registered_model_name,
                    r.registered_model_version,
                    r.export_method,
                    r.exporter_module,
                )
            return

        run_logger.warning(
            "Approach '%s' has no attribute 'model' or 'models' to log.",
            approach_name,
        )

    except Exception as e:
        run_logger.warning("Failed to log model for %s: %s", approach_name, e)
