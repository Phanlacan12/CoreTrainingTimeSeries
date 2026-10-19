import json
from typing import Any, Dict, Optional

import pandas as pd

from tsfb.base.utils.utils import load_config


def infer_direction(metric_name: str) -> str:
    """Guess whether 'higher' or 'lower' values are better based on metric name."""
    lower = ["rmse", "mse", "mae", "mape", "smape"]
    higher = ["r2"]
    m = metric_name.lower()
    if any(h in m for h in higher):
        return "higher"
    if any(h in m for h in lower):
        return "lower"
    return "lower"


def _read_yaml(path: str) -> Dict[str, Any]:
    """Read YAML file into a Python dictionary."""
    return load_config(path)


def _find_model_block_in_config(
    cfg: Dict[str, Any], model_name: str
) -> Optional[Dict[str, Any]]:
    """Find model configuration block by name inside a YAML config."""
    models = cfg.get("models")
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model_name") or m.get("id")
                if n == model_name:
                    return m
    for k in ["model", "models", "estimator"]:
        blk = cfg.get(k)
        if isinstance(blk, dict):
            # Check if model_name is a key
            if model_name in blk and isinstance(blk[model_name], dict):
                return {"name": model_name, **blk[model_name]}
            # Check if blk itself defines the model_name
            n = blk.get("name") or blk.get("model_name")
            if n == model_name:
                return blk
    return None


def _parse_json_field(val):
    """Safely parse JSON-like values (returns None if empty/NaN)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return s


def select_best_model(
    df: pd.DataFrame,
    config_path: str,
    metric: str,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Select the best model from results DataFrame by metric."""
    metric_col = next((c for c in df.columns if c.lower() == metric.lower()), None)
    if metric_col is None:
        raise ValueError(f"Metric '{metric}' not found in results file (case-ins).")

    direction = infer_direction(metric) if mode == "auto" else mode.lower()
    if direction not in {"higher", "lower"}:
        raise ValueError("mode must be 'auto', 'higher' or 'lower'.")

    if df[metric_col].isna().all():
        raise ValueError(
            f"All values for metric '{metric_col}' are NaN. Cannot select best model."
        )

    best_row = (
        df.loc[df[metric_col].idxmax()]
        if direction == "higher"
        else df.loc[df[metric_col].idxmin()]
    )
    best_model = str(best_row["approach_name"])
    best_score = float(best_row[metric_col])

    strategy_args = (
        _parse_json_field(best_row.get("strategy_args"))
        if "strategy_args" in df.columns
        else None
    )
    model_params = (
        _parse_json_field(best_row.get("model_params"))
        if "model_params" in df.columns
        else None
    )

    cfg = _read_yaml(config_path)
    best_block = _find_model_block_in_config(cfg, best_model) or {"name": best_model}

    return {
        "config": best_block,
        "name": best_model,
        "metric": metric_col,
        "score": best_score,
        "strategy_args": strategy_args,
        "model_params": model_params,
    }
