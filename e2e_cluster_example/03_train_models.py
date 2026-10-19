"""Train one model per cluster and track them via the MLflow Model Registry.

Current tsfb logging mechanism (see tsfb/base/mlflow/):
  - Forecast model: ONNX preferred (fallback PyFunc), registered as
        <experiment_name>_<approach_name>
  - Preprocess model: PreprocessPyFunc, registered as
        <experiment_name>_<approach_name>_preprocess

So instead of tracking run IDs and crawling run artifacts, we only need to
record the registered model names per cluster. The serving app then loads
them with `models:/<name>/<version>` URIs.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from tsfb.base.mlflow.mlflow_master import MLflowMaster
from tsfb.base.mlflow.model_logging import clean_name
from tsfb.run_pipeline import PipelineExecutor


def _drop_timezone_from_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp-like columns to timezone-naive datetimes."""
    result = df.copy()
    time_cols = [
        col
        for col in result.columns
        if "timestamp" in str(col).lower() or str(col).lower() in {"date", "ds"}
    ]

    for col in time_cols:
        try:
            dt_series = pd.to_datetime(result[col])
        except (TypeError, ValueError):
            continue
        if not pd.api.types.is_datetime64_any_dtype(dt_series):
            continue
        if getattr(dt_series.dt, "tz", None) is not None:
            dt_series = dt_series.dt.tz_convert(None)
        result[col] = dt_series

    return result


def prepare_data():
    try:
        cluster_df = pd.read_csv("data/cluster_mapping.csv")
        clusters = cluster_df["cluster_id"].unique()
        has_mapping = True
    except FileNotFoundError:
        print(
            "data/cluster_mapping.csv not found. Falling back to single 'default_cluster' for all data."
        )
        clusters = ["default_cluster"]
        has_mapping = False

    try:
        data_df = pd.read_csv("data/offline_features.csv")
    except FileNotFoundError:
        data_df = pd.read_csv("data/epidemic_cases_ts.csv")

    for cluster in clusters:
        if has_mapping:
            series_in_cluster = cluster_df[cluster_df["cluster_id"] == cluster][
                "series_id"
            ].tolist()
            cluster_data = data_df[data_df["series_id"].isin(series_in_cluster)].copy()
        else:
            cluster_data = data_df.copy()

        cluster_data = _drop_timezone_from_time_columns(cluster_data)

        data_path = f"data/{cluster}_data.csv"
        os.makedirs("data", exist_ok=True)
        cluster_data.to_csv(data_path, index=False)
        print(f"Data for {cluster} saved to {data_path} with {len(cluster_data)} rows.")

    return clusters


def registered_model_names_from_config(cfg_path: str) -> dict:
    """Derive the registered model names produced by the tsfb pipeline.

    Mirrors tsfb.base.mlflow.approach_handle.log_single_approach_model and
    ForecastingStrategy._maybe_log_preprocess_model naming.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get("mlflow", {}).get(
        "experiment_name", "ForecastingBenchmark"
    )
    approach_name = next(iter(cfg.get("model", {})))

    forecast_name = f"{clean_name(experiment_name)}_{clean_name(approach_name)}"
    return {
        "experiment_name": experiment_name,
        "forecast": forecast_name,
        "preprocess": f"{clean_name(experiment_name)}_{clean_name(approach_name)}_preprocess",
    }


def resolve_latest_version(client: MlflowClient, name: str):
    """Return the latest registered version of a model, or None if absent."""
    versions = client.search_model_versions(f"name = '{name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def train_and_log():
    clusters = prepare_data()
    cluster_model_map = {}

    tracking_uri = os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # PipelineExecutor runs the same health check internally and gracefully
    # degrades (training continues without MLflow). We check up-front so the
    # example fails fast: without MLflow there is nothing to serve in step 04.
    if not MLflowMaster.check_health(tracking_uri, timeout=10, retry_interval=2):
        raise RuntimeError(
            f"MLflow tracking server '{tracking_uri}' is not reachable. "
            "The serving step needs the Model Registry, aborting."
        )

    client = MlflowClient()

    for cluster in clusters:
        cfg_path = f"configs/config_{cluster}.yaml"
        if not os.path.exists(cfg_path):
            print(f"Config {cfg_path} not found. Skipping...")
            continue

        print(f"\n=== Training model for {cluster} using standalone config ===")
        executor = PipelineExecutor(
            config_path=cfg_path, results_path=f"data/uc7_results_{cluster}.csv"
        )
        executor.run()

        # The pipeline registered the models; resolve their latest versions.
        names = registered_model_names_from_config(cfg_path)
        entry = {}
        for role in ("forecast", "preprocess"):
            name = names[role]
            latest = resolve_latest_version(client, name)
            if latest is None:
                print(f"  WARNING: registered model '{name}' not found for {cluster}.")
                continue
            entry[role] = {
                "registered_model_name": name,
                "version": latest.version,
                "model_uri": f"models:/{name}/{latest.version}",
            }
            print(
                f"  Tracked {role} model for {cluster}: "
                f"models:/{name}/{latest.version}"
            )

        if entry:
            cluster_model_map[cluster] = entry

    with open("data/cluster_model_map.json", "w") as f:
        json.dump(cluster_model_map, f, indent=2)

    print(
        "\nTraining finished. Model registry map saved to data/cluster_model_map.json"
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_and_log()
