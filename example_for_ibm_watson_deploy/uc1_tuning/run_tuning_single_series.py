import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfb.base.data_module.factory import DataPipelineFactory
from tsfb.base.utils.utils import load_config
from tsfb.tuning.base_tuning import BaseTuning
from tsfb.tuning.schema import TuningConfig

# By default, load the configuration used for tuning
CONFIG_PATH = (
    REPO_ROOT
    / "example_for_ibm_watson_deploy/uc5_tuning/tuning_single_series_config.yaml"
)


def set_env_vars_from_config(cfg: dict):
    """Set environment variables like SingleStore credentials from config."""
    if "env_vars" in cfg:
        for k, v in cfg["env_vars"].items():
            os.environ[str(k)] = str(v)


def run_single_series_tuning():
    print("\n--- Running Single Series Tuning ---")
    os.chdir(REPO_ROOT)

    # 1. Load Config
    print(f"Loading config from: {CONFIG_PATH}")
    try:
        full_config = load_config(str(CONFIG_PATH))
    except Exception:
        with open(CONFIG_PATH, "r") as f:
            full_config = yaml.safe_load(f)

    # Set external Env Vars for SingleStore and MLflow
    set_env_vars_from_config(full_config)

    # 2. Setup Custom MLflow
    tracking_uri = full_config.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if full_config.get("mlflow", {}).get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = full_config["mlflow"]["tracking_uri"]
        if (
            "env_vars" not in full_config
            or "MLFLOW_S3_ENDPOINT_URL" not in full_config["env_vars"]
        ):
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.218:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "admin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "Dgs@032026"

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = full_config.get("mlflow", {}).get(
        "experiment_name", "Single_Series_Tuning_Experiment"
    )
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")

    # 3. Handle DataModule for SingleStore
    # We locate data config from the same pattern as uc7_mlflow
    data_cfg = (
        full_config.get("evaluation", {})
        .get("strategy_args", {})
        .get("data_module_config", {})
    )
    if not data_cfg and "data_module_config" in full_config:
        data_cfg = full_config["data_module_config"]

    if not data_cfg:
        raise ValueError("Cannot find 'data_module_config' in the configuration.")

    print(
        f"Reading data config: {data_cfg.get('kind', 'unknown')} | {data_cfg.get('file_path', '')}"
    )

    # Build Dataloader Factory
    reader, preprocess, dataloader = DataPipelineFactory.create(data_cfg)

    # Load raw data (this handles querying SingleStore or reading CSV based on config)
    (raw_target, raw_past_cov, raw_future_cov, raw_static_cov) = reader.load_data()
    df = raw_target

    print(f"Data Loaded: {df.shape}")

    # Extract horizon and target info
    horizon = (
        full_config.get("evaluation", {}).get("strategy_args", {}).get("horizon", 10)
    )
    target_columns = data_cfg.get("target_columns", [])
    target_col = target_columns[0] if target_columns else df.columns[-1]

    # Check to ensure horizon is valid compared to df shape
    if len(df) <= horizon:
        raise ValueError(
            f"Data length ({len(df)}) is less than or equal to horizon ({horizon})!"
        )

    # Keep only the target column so the model treats it as a single-target problem
    lookback_data = df.iloc[:-horizon][[target_col]]
    y_true = df.iloc[-horizon:][target_col].values

    # 4. Prepare Tuning Configs
    tuning_section = full_config.get(
        "tuning",
        {
            # Fallback tuning config if not present in the new yaml
            "n_trials": 5,
            "metric": "rmse",
            "metrics_eval": ["rmse"],
            "direction": "minimize",
            "parameters": [
                {
                    "name": "input_chunk_length",
                    "type": "categorical",
                    "choices": [24, 48, 96, 192],
                },
                {
                    "name": "learning_rate",
                    "type": "float",
                    "min": 1.0e-5,
                    "max": 1.0e-1,
                    "log": True,
                },
                {"name": "batch_size", "type": "categorical", "choices": [16, 32, 64]},
            ],
        },
    )
    tuning_config = TuningConfig(**tuning_section)
    print(
        f"Tuning objective metric: {tuning_config.metric} | "
        f"evaluation metrics: {tuning_config.metrics_eval}"
    )

    model_key = list(full_config.get("model", {}).keys())[0]
    base_model_config = full_config["model"][model_key]
    model_name = base_model_config.get("model_name", "darts.models.NLinearModel")

    # 5. Run Tuning
    try:
        from tsfb.base.utils.parallel import ParallelBackend

        ParallelBackend().close(force=True)
    except:
        pass

    safe_model_name = model_name.replace(".", "_")
    tuner = BaseTuning(
        study_name=f"single_series_study_debug_{safe_model_name}",
        direction=tuning_config.direction,
        storage="sqlite:///single_tuning.db",
    )

    # Extract custom tags from mlflow config section
    mlflow_tags = full_config.get("mlflow", {}).get("tags", {})
    if mlflow_tags:
        print(f"Custom MLflow tags from config: {list(mlflow_tags.keys())}")

    study = tuner.tune_single_series(
        model_name=model_name,
        base_model_config=base_model_config,
        tuning_config=tuning_config,
        y_true=y_true,
        lookback_data=lookback_data,
        horizon=horizon,
        log_to_mlflow=True,  # Log top K results to MLflow
        custom_tags=mlflow_tags if mlflow_tags else None,
    )

    try:
        print(f"Single Series Best Params: {study.best_params}")
        print(f"Single Series Best Value: {study.best_value}")
    except ValueError:
        print("Tuning finished but no successful trials were completed.")


if __name__ == "__main__":
    os.makedirs("example_output", exist_ok=True)
    run_single_series_tuning()
