import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from numpy.typing import DTypeLike

from tsfb.base.mlflow.mlflow_master import MLflowMaster
from tsfb.base.utils.logging import get_logger, setup_logging
from tsfb.run_pipeline import PipelineExecutor

# Configure logger for easy tracking
setup_logging(lvl="INFO", trace=False)
logger = get_logger("uc7_download_serve")


def inject_hardcoded_configs(cfg: dict) -> dict:
    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    if "strategy_args" not in cfg["evaluation"]:
        cfg["evaluation"]["strategy_args"] = {}

    cfg["evaluation"]["series_list"] = ["__default__"]

    cfg["evaluation"]["strategy_args"].update(
        {
            "strategy_name": "fixed_forecast",
            "save_true_pred": True,
            "seed": 42,
            "deterministic": "full",
            "target_channel": {"__default__": None},
        }
    )
    return cfg


def np_dtype_from_onnx(onnx_type: str) -> DTypeLike:
    """Convert ONNX tensor type string to a NumPy-compatible dtype-like object."""
    t = (onnx_type or "").lower()
    mapping = {
        "double": np.float64,
        "float16": np.float16,
        "float": np.float32,
        "int64": np.int64,
        "int32": np.int32,
        "bool": np.bool_,
    }
    for key, np_type in mapping.items():
        if key in t:
            return np_type
    return np.float32


def setup_mlflow(config_path: str) -> Tuple[dict, str]:
    """
    Load MLflow DB/Storage configuration from the config yaml (inherited from mlflow_example.py)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = inject_hardcoded_configs(cfg)

    # Get tracking URI from config, fallback to environment variable if absent
    tracking_uri = cfg.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if tracking_uri and not tracking_uri.startswith(("sqlite", "file")):
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    return cfg, experiment_name


def run_training_pipeline(config_path: str) -> None:
    logger.info(
        "===================== [1] START TRAINING & MLFLOW LOGGING ====================="
    )

    cfg, _ = setup_mlflow(config_path)

    executor = PipelineExecutor(
        config_path=config_path, results_path="uc7_mlflow_results.csv"
    )

    original_load_config = executor._load_config

    def _patched_load_config():
        original_load_config()
        executor.cfg = inject_hardcoded_configs(executor.cfg)

    executor._load_config = _patched_load_config
    executor.run()
    logger.info("Pipeline executed successfully. Model logged to MLflow.")


def get_latest_parent_run_id(experiment_name: str) -> str:
    logger.info(
        "===================== [2] FIND PARENT RUN (pipeline_run) ====================="
    )
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Find the latest Parent Run named 'pipeline_run'
    parent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'pipeline_run'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not parent_runs:
        raise ValueError("Could not find any 'pipeline_run' in this experiment.")

    parent_run_id = parent_runs[0].info.run_id
    logger.info(f"Found latest Parent Run ID: {parent_run_id}")
    return parent_run_id


def download_artifacts_from_child_runs(
    parent_run_id: str, experiment_name: str, dest_dir: str
) -> Tuple[str, str, Optional[str]]:
    logger.info(
        "===================== [3] DOWNLOAD ARTIFACTS FROM CHILD RUNS TO LOCAL ====================="
    )
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    # Find all Child Runs based on the parentRunId tag
    child_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
    )

    if not child_runs:
        raise ValueError(f"No child runs found for parent run {parent_run_id}")

    # In this example, we take the first model (usually the only model in the UC7 yaml file)
    child_run = child_runs[0]
    child_run_id = child_run.info.run_id
    child_run_name = child_run.data.tags.get("mlflow.runName", "unknown_model")

    logger.info(f"Selected Sub Run: {child_run_name} (ID: {child_run_id})")

    # Artifact URIs present in the child run
    onnx_model_uri = f"runs:/{child_run_id}/{child_run_name}"
    preprocess_model_uri = f"runs:/{child_run_id}/{child_run_name}_preprocess"

    # Local storage directory (dest_dir defaults to ./local_serving_models)
    os.makedirs(dest_dir, exist_ok=True)

    # 3.1 Download ONNX Model Folder (Automatically expands into 'dest_dir/child_run_name')
    logger.info(f"Downloading ONNX artifact to {dest_dir} ...")
    local_onnx_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=onnx_model_uri, dst_path=dest_dir
    )

    # 3.2 Download Preprocess Model Folder (Automatically expands into 'dest_dir/child_run_name_preprocess')
    local_prep_dir = None
    try:
        logger.info(f"Downloading Preprocess artifact to {dest_dir} ...")
        local_prep_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=preprocess_model_uri, dst_path=dest_dir
        )
    except Exception as e:
        logger.warning(f"No preprocess model found or could not download: {e}")

    return child_run_name, local_onnx_dir, local_prep_dir


def load_local_models(
    local_onnx_dir: str, local_prep_dir: Optional[str]
) -> Tuple[Any, Any]:
    logger.info(
        "===================== [4] LOAD MODELS FROM LOCAL FOLDERS ====================="
    )

    # Load Preprocess model (.pkl)
    preprocess_model = None
    if local_prep_dir and os.path.exists(local_prep_dir):
        logger.info(f"Loading preprocess model from LOCAL directory: {local_prep_dir}")
        preprocess_model = mlflow.pyfunc.load_model(local_prep_dir)
    else:
        logger.warning("No local preprocess model to load.")

    # Find the model.onnx file in the downloaded local directory
    onnx_file_path = os.path.join(local_onnx_dir, "model.onnx")
    if not os.path.exists(onnx_file_path):
        onnx_files = list(Path(local_onnx_dir).rglob("*.onnx"))
        if onnx_files:
            onnx_file_path = onnx_files[0].as_posix()
        else:
            raise FileNotFoundError(f"Cannot find any .onnx file in {local_onnx_dir}")

    logger.info(f"Loading ONNX model from LOCAL file: {onnx_file_path}")
    sess = ort.InferenceSession(onnx_file_path, providers=ort.get_available_providers())

    for inp in sess.get_inputs():
        logger.info(f"ONNX input: name={inp.name}, shape={inp.shape}, dtype={inp.type}")

    return sess, preprocess_model


def generate_dummy_data(cfg: dict, model_name_cfg: str) -> pd.DataFrame:
    logger.info(
        "===================== [5] SERVING EVALUATION (DUMMY DATA) ====================="
    )

    input_chunk_length = cfg["model"][model_name_cfg]["model_hyper_params"].get(
        "input_chunk_length", 24
    )
    target_columns = cfg.get("data_module_config", {}).get("target_columns", ["OT"])

    # Generate dummy data
    index = pd.date_range("2024-01-01", periods=input_chunk_length, freq="h")
    raw_data = pd.DataFrame(
        {col: np.random.rand(input_chunk_length) for col in target_columns},
        index=index,
    )
    return raw_data


def serve_local_models(
    sess: Any, preprocess_model: Any, raw_data: pd.DataFrame, cfg: dict
) -> np.ndarray:
    logger.info(
        "===================== [6] PREPROCESS USING LOCAL MODEL ====================="
    )

    target_columns = cfg.get("data_module_config", {}).get("target_columns", ["OT"])

    if preprocess_model:
        prep_out = preprocess_model.predict({"target": raw_data})
        target_proc = prep_out["target_processed"]
        logger.info("Preprocessed raw data locally successfully.")
    else:
        target_proc = raw_data

    available_cols = [c for c in target_columns if c in target_proc.columns]
    if not available_cols:
        available_cols = list(target_proc.columns)

    model_name_cfg = list(cfg["model"].keys())[0]
    input_chunk_length = cfg["model"][model_name_cfg]["model_hyper_params"].get(
        "input_chunk_length", 24
    )

    x_past = (
        target_proc[available_cols].iloc[-input_chunk_length:].to_numpy()[None, :, :]
    )

    feeds = {}
    for inp in sess.get_inputs():
        dtype = np_dtype_from_onnx(inp.type)
        if inp.name == "x_past":
            feeds[inp.name] = x_past.astype(dtype)
        else:
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            feeds[inp.name] = np.zeros(shape, dtype=dtype)

    logger.info(f"ONNX feeds ready: { {k: v.shape for k, v in feeds.items()} }")

    logger.info("===================== [7] RUN ONNX INFERENCE =====================")

    yhat = sess.run(None, feeds)[0]
    if yhat.ndim == 3 and yhat.shape[0] == 1:
        yhat = yhat[0]

    logger.info(f"Forecast from local model completed! Output shape: {yhat.shape}")
    logger.info(f"Forecast sample (first 5): {yhat.flatten()[:5].tolist()}")
    return yhat


def main():
    # YAML configuration file
    CONFIG_PATH = "example_for_ibm_watson_deploy/uc7_mlflow/configs/mlflow_config.yaml"
    LOCAL_DEST_DIR = "./local_serving_models"

    cfg, experiment_name = setup_mlflow(CONFIG_PATH)

    # 1. Run training pipeline and save results to MLflow Server
    run_training_pipeline(CONFIG_PATH)
    time.sleep(2)  # Wait briefly to ensure S3 / MLflow DB logs are fully synchronized

    # 2. Clear model folder cache (if any, to visually demonstrate a fresh download)
    if os.path.exists(LOCAL_DEST_DIR):
        shutil.rmtree(LOCAL_DEST_DIR)

    # 3. Get Parent Run ID
    parent_run_id = get_latest_parent_run_id(experiment_name)

    # 4. Download Artifacts to local folder
    child_run_name, local_onnx_dir, local_prep_dir = download_artifacts_from_child_runs(
        parent_run_id=parent_run_id,
        experiment_name=experiment_name,
        dest_dir=LOCAL_DEST_DIR,
    )

    # 5. Load model from Local Folder
    sess, preprocess_model = load_local_models(local_onnx_dir, local_prep_dir)

    # 6. Prepare data and serve forecast
    model_name_cfg = list(cfg["model"].keys())[0]
    raw_data = generate_dummy_data(cfg, model_name_cfg)

    predictions = serve_local_models(sess, preprocess_model, raw_data, cfg)
    logger.info(
        "SUCCESS!! Entire flow complete: Train -> Run ID -> Pull Locale -> Serve Locally."
    )


if __name__ == "__main__":
    main()
