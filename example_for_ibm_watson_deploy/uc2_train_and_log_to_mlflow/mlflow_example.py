import os
import shutil
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import yaml
from numpy.typing import DTypeLike

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.mlflow.mlflow_master import MLflowMaster
from tsfb.base.utils.logging import get_logger, setup_logging
from tsfb.base.utils.utils import load_config
from tsfb.reduce_size import BaseReduceSize, ReduceSizeConfig
from tsfb.run_pipeline import PipelineExecutor

# Configure logger
setup_logging(lvl="INFO", trace=False)
logger = get_logger("uc7_mlflow_example")


class OldONNXApproach(ForecastingApproach):
    """
    Wraps an already-logged MLflow ONNX model to evaluate alongside newly trained models.
    """

    def __init__(self, run_id: str, model_name: str, config: dict):
        super().__init__()
        self.name = f"old_model_{run_id}"
        self.config = config.copy()

        # Users provide the child run ID where the artifacts actually live
        logger.info(f"Pulling OLD artifacts from CHILD run ID: {run_id} ...")

        dest_dir = "./local_serving_models"
        self.old_config = {}

        try:
            logger.info("Downloading ROOT artifacts to find config yaml...")
            run_root_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/", dst_path=dest_dir
            )
            yaml_files = list(Path(run_root_dir).glob("*.yaml"))
            if yaml_files:
                with open(yaml_files[0], "r") as f:
                    self.old_config = yaml.safe_load(f)
                logger.info(f"Loaded old config from {yaml_files[0]}")
        except Exception as e:
            logger.warning(f"Could not load old config yaml: {e}")

        os.makedirs(dest_dir, exist_ok=True)

        onnx_model_uri = f"runs:/{run_id}/{model_name}"
        preprocess_model_uri = f"runs:/{run_id}/{model_name}_preprocess"

        logger.info(
            f"Downloading ONNX artifact from {onnx_model_uri} to {dest_dir} ..."
        )
        local_onnx_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=onnx_model_uri, dst_path=dest_dir
        )

        logger.info(
            f"Downloading Preprocess artifact from {preprocess_model_uri} to {dest_dir} ..."
        )
        try:
            local_prep_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=preprocess_model_uri, dst_path=dest_dir
            )
            logger.info(
                f"Loading OLD preprocess pyfunc from LOCAL dir: {local_prep_dir}"
            )
            self.preprocess_model = mlflow.pyfunc.load_model(local_prep_dir)
            logger.info("Successfully loaded OLD preprocess pyfunc model.")
        except Exception as e:
            logger.warning(f"Could not load old preprocess model: {e}")
            self.preprocess_model = None

        # Temp dir needs to be kept alive as long as InferenceSession lives
        # Find exactly the model.onnx file in the downloaded dir
        onnx_file_path = os.path.join(local_onnx_dir, "model.onnx")
        if not os.path.exists(onnx_file_path):
            onnx_files = list(Path(local_onnx_dir).rglob("*.onnx"))
            if onnx_files:
                onnx_file_path = onnx_files[0].as_posix()

        logger.info(f"Loading ONNX model from LOCAL file: {onnx_file_path}")
        self.sess = ort.InferenceSession(
            str(onnx_file_path), providers=ort.get_available_providers()
        )
        logger.info("Loaded old ONNX model into InferenceSession.")

    def forecast_fit(self, train_data, **kwargs):
        # Model is already trained
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        expected_seq_len = self.config.get("model_hyper_params", {}).get(
            "input_chunk_length", 24
        )

        if self.old_config and "model" in self.old_config:
            for k, v in self.old_config["model"].items():
                if isinstance(v, dict) and "model_hyper_params" in v:
                    if "input_chunk_length" in v["model_hyper_params"]:
                        expected_seq_len = v["model_hyper_params"]["input_chunk_length"]
                        break

        # Note: `lookback_data` inside TSFB pipeline's `eval_model` is `norm_train_df`
        # which means it has ALREADY been preprocessed by the NEW model's pipeline scaler!
        # Because we assume the new run and the old run share the exact same configuration logic (same scale logic),
        # we DO NOT need to apply `self.preprocess_model` again!
        # Applying it here on already-scaled data would corrupt the values.
        # We simply extract the last `expected_seq_len` window and feed it to ONNX.
        x_past = lookback_data.iloc[-expected_seq_len:].to_numpy()[None, :, :]

        feeds = {}
        for inp in self.sess.get_inputs():
            dtype = np_dtype_from_onnx(inp.type)
            if inp.name == "x_past":
                feeds[inp.name] = x_past.astype(dtype)
            else:
                shape = [s if isinstance(s, int) else 1 for s in inp.shape]
                feeds[inp.name] = np.zeros(shape, dtype=dtype)

        yhat = self.sess.run(None, feeds)[0]

        # Safely reduce dimensions to 2D (horizon, n_vars)
        if yhat.ndim > 0 and yhat.shape[0] == 1:
            yhat = yhat[0]
        if yhat.ndim == 3 and yhat.shape[-1] == 1:
            yhat = yhat[:, :, 0]

        columns = (
            lookback_data.columns
            if lookback_data.shape[1] == yhat.shape[1]
            else [self.name] * yhat.shape[1]
        )

        if isinstance(lookback_data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(lookback_data.index) or "H"
            future_index = pd.date_range(
                lookback_data.index[-1], periods=len(yhat) + 1, freq=freq
            )[1:]
        else:
            future_index = np.arange(len(lookback_data), len(lookback_data) + len(yhat))

        return pd.DataFrame(yhat, index=future_index, columns=columns)

    def batch_forecast(self, horizon, batch_maker, covariates=None, **kwargs):
        raise NotImplementedError("Batch forecast not implemented for OldONNXApproach")


def inject_hardcoded_configs(cfg: dict):
    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    if "strategy_args" not in cfg["evaluation"]:
        cfg["evaluation"]["strategy_args"] = {}

    cfg["evaluation"]["series_list"] = ["__default__"]

    cfg["evaluation"]["strategy_args"].update(
        {
            "strategy_name": "fixed_forecast",
            # "stride": 1,
            # "num_rollings": 1000,
            "save_true_pred": True,
            "seed": 42,
            "deterministic": "full",
            "target_channel": {"__default__": None},
        }
    )
    return cfg


def set_env_vars_from_config(cfg: dict):
    if "env_vars" in cfg:
        for k, v in cfg["env_vars"].items():
            os.environ[str(k)] = str(v)


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


ONNX_REQUIRED_MODELS = [
    "TCNModel",
    "TFTModel",
    "TransformerModel",
    "NHiTSModel",
    "TiDEModel",
    "BlockRNNModel",
    "RNNModel",
    "DLinearModel",
    "NBEATSModel",
    "NLinearModel",
]


def validate_onnx_export(executor, captured_approaches):
    """
    Check if the trained model requires an ONNX file (e.g., TFT, DLinear, NLinear).
    If it requires an ONNX file but none was found in artifacts, update the run status to 'FAILED'.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_name = executor.cfg.get("mlflow", {}).get(
            "experiment_name", "ForecastingBenchmark"
        )
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:
            for approach_name, approach in captured_approaches.items():
                model_obj = getattr(approach, "model", None)
                if not model_obj:
                    continue

                model_class = type(model_obj).__name__
                if (
                    hasattr(model_obj, "model")
                    and getattr(model_obj, "model") is not None
                ):
                    model_class = type(getattr(model_obj, "model")).__name__

                needs_onnx = model_class in ONNX_REQUIRED_MODELS
                if not needs_onnx:
                    continue

                logger.info(
                    f"Checking ONNX artifact for required model: {model_class}_{approach_name}"
                )

                is_single = len(captured_approaches) == 1
                flat_name = f"{experiment_name}_{approach_name}"
                run_filter = (
                    f"run_name = '{flat_name}'"
                    if is_single
                    else f"run_name = '{approach_name}'"
                )

                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=run_filter,
                    order_by=["start_time DESC"],
                    max_results=1,
                )

                if runs:
                    run_id = runs[0].info.run_id

                    def has_onnx_file(r_id, path=""):
                        try:
                            for file_info in client.list_artifacts(r_id, path):
                                if file_info.is_dir:
                                    if has_onnx_file(r_id, file_info.path):
                                        return True
                                elif file_info.path.endswith(".onnx"):
                                    return True
                        except Exception:
                            pass
                        return False

                    if not has_onnx_file(run_id):
                        logger.error(
                            f"Missing ONNX file in artifacts for {model_class} (Run {run_id}). Updating status to FAILED."
                        )
                        client.set_terminated(run_id, status="FAILED")
                    else:
                        logger.info(
                            f"ONNX file verification passed for {model_class} in Run {run_id}."
                        )
    except Exception as e:
        logger.warning(f"Failed to validate ONNX file status for MLflow runs: {e}")


def run_training_and_logging(config_path: str):
    """
    Step 1: Parse the configuration, train the model, log it to MLflow,
    and apply reduce_size (quantization) if configured.
    """
    logger.info(
        "===================== [1] START TRAINING & MLFLOW LOGGING ====================="
    )

    # Set env variables before initializing Pipeline
    cfg = load_config(config_path)
    set_env_vars_from_config(cfg)
    tracking_uri = cfg.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if cfg.get("mlflow", {}).get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
        if "env_vars" not in cfg or "MLFLOW_S3_ENDPOINT_URL" not in cfg["env_vars"]:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.218:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "admin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "Dgs@032026"

    executor = PipelineExecutor(
        config_path=config_path, results_path="uc7_mlflow_results.csv"
    )

    original_load_config = executor._load_config

    def _patched_load_config():
        original_load_config()
        executor.cfg = inject_hardcoded_configs(executor.cfg)
        if "mlflow" not in executor.cfg:
            executor.cfg["mlflow"] = {}
        executor.cfg["mlflow"]["register_model"] = False

    executor._load_config = _patched_load_config

    # Monkey-patch _run_evaluation_and_save_results to capture trained approaches
    captured_approaches = {}
    original_run_eval = executor._run_evaluation_and_save_results

    def _patched_run_eval(approaches):
        captured_approaches.update(approaches)
        return original_run_eval(approaches)

    executor._run_evaluation_and_save_results = _patched_run_eval

    # Run the pipeline (which internally logs to MLflow if enabled in config)
    executor.run()
    logger.info("Pipeline executed successfully. Model logged to MLflow.")

    validate_onnx_export(executor, captured_approaches)

    # ── UC8: Apply reduce_size (quantization) after training ──
    if "reduce_size_config" in executor.cfg and captured_approaches:
        _apply_reduce_size(executor.cfg, captured_approaches)


def evaluate_old_model_from_mlflow(config_path: str, old_run_id: str):
    """
    Separated step 1b: Evaluate Old ONNX Model using original data processing pipeline.
    After evaluation, log the old model's metrics into the new model's latest
    MLflow run with a 'baseline_' prefix for direct comparison.
    """
    logger.info(
        "===================== [1B] EVALUATE OLD MODEL ON TEST SPLIT ====================="
    )

    cfg = load_config(config_path)
    set_env_vars_from_config(cfg)
    tracking_uri = cfg.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if cfg.get("mlflow", {}).get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
        if "env_vars" not in cfg or "MLFLOW_S3_ENDPOINT_URL" not in cfg["env_vars"]:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.218:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "admin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "Dgs@032026"

    executor = PipelineExecutor(
        config_path=config_path, results_path="uc7_mlflow_results.csv"
    )

    original_load_config = executor._load_config

    def _patched_load_config():
        original_load_config()
        executor.cfg = inject_hardcoded_configs(executor.cfg)
        if "mlflow" in executor.cfg:
            executor.cfg["mlflow"]["enabled"] = False

    executor._load_config = _patched_load_config

    # Force PipelineExecutor to run ONLY the OldONNXApproach on the same data configuration
    def _patched_build_approaches():
        model_name = list(executor.cfg["model"].keys())[0]
        experiment_name = executor.cfg.get("mlflow", {}).get(
            "experiment_name", "Default_Experiment"
        )
        # Flattened structure: {experiment_name}_{model_name}
        flat_model_name = f"{experiment_name}_{model_name}"

        old_app = OldONNXApproach(
            old_run_id, flat_model_name, executor.cfg["model"][model_name]
        )
        return {old_app.name: old_app}

    executor._build_approaches = _patched_build_approaches

    # Run the evaluation
    executor.run()
    logger.info(
        f"Evaluation of old_model {old_run_id} completed. Appended to uc7_mlflow_results.csv."
    )

    # ── Log old model metrics into the new model's run as baseline_ ──
    _log_baseline_metrics_to_new_run(cfg, executor.results_path, old_run_id)


def _log_baseline_metrics_to_new_run(
    cfg: dict, results_path: str, old_run_id: str
) -> None:
    """Read old model evaluation results and log them as baseline metrics
    into the new model's latest MLflow child run.

    Args:
        cfg: Full configuration dictionary.
        results_path: Path to the CSV containing evaluation results.
        old_run_id: The run ID of the old model (used to filter results).
    """
    try:
        results_df = pd.read_csv(results_path)
    except Exception as e:
        logger.warning(f"Could not read results for baseline logging: {e}")
        return

    # Filter rows belonging to the old model
    old_model_prefix = f"old_model_{old_run_id}"
    if "approach_name" in results_df.columns:
        old_rows = results_df[
            results_df["approach_name"].str.contains(old_model_prefix, na=False)
        ]
    else:
        logger.warning(
            "No 'approach_name' column found in results. Skipping baseline logging."
        )
        return

    if old_rows.empty:
        logger.warning(
            f"No results found for old model '{old_model_prefix}'. Skipping baseline logging."
        )
        return

    # Compute mean metrics for old model
    numeric_cols = [
        c for c in old_rows.columns if pd.api.types.is_numeric_dtype(old_rows[c])
    ]
    old_metrics = old_rows[numeric_cols].mean().to_dict()

    if not old_metrics:
        logger.warning("No numeric metrics found for old model.")
        return

    # Find the new model's latest child run in MLflow
    experiment_name = cfg.get("mlflow", {}).get("experiment_name")
    if not experiment_name:
        logger.warning("No experiment_name in config. Cannot log baseline metrics.")
        return

    model_name = list(cfg.get("model", {}).keys())[0]
    flat_model_name = f"{experiment_name}_{model_name}"

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.warning(f"Experiment '{experiment_name}' not found. Skipping baseline.")
        return

    # Search for the latest child run matching the new model's flat approach name
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_name = '{flat_model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        logger.warning(
            f"No MLflow run named '{flat_model_name}' found. Skipping baseline logging."
        )
        return

    new_run_id = runs[0].info.run_id
    logger.info(
        f"Logging baseline metrics from old model into new model run "
        f"'{flat_model_name}' (run_id={new_run_id})..."
    )

    # Log each old metric with baseline_ prefix into the new model's run
    for metric_name, value in old_metrics.items():
        baseline_key = f"baseline_{old_run_id}_{metric_name}"
        try:
            client.log_metric(new_run_id, baseline_key, float(value))
        except Exception as e:
            logger.warning(f"Failed to log baseline metric '{baseline_key}': {e}")

    logger.info(
        f"Logged {len(old_metrics)} baseline metrics into run {new_run_id}: "
        f"{[f'baseline_{old_run_id}_{k}' for k in old_metrics.keys()]}"
    )


def _apply_reduce_size(cfg: dict, approaches: dict):
    """
    Apply UC8 reduce_size (quantization) on the trained models, evaluate the reduced model,
    and log reduced artifacts along with metrics to MLflow natively in a flat run.
    """
    logger.info(
        "===================== [UC8] REDUCE SIZE, EVAL & LOG ====================="
    )
    from tsfb.base.evaluation.evaluate_model import eval_model

    if "reduce_size_config" not in cfg:
        logger.warning("No 'reduce_size_config' provided in configuration.")
        return

    reduce_cfg = ReduceSizeConfig.from_dict(cfg["reduce_size_config"])
    logger.info(
        f"Reduce size config: method={reduce_cfg.method}, "
        f"target_ratio={reduce_cfg.target_ratio}, "
        f"output_dir={reduce_cfg.output_dir}"
    )

    tracking_uri = cfg.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if cfg.get("mlflow", {}).get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
        if "env_vars" not in cfg or "MLFLOW_S3_ENDPOINT_URL" not in cfg["env_vars"]:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.218:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "admin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "Dgs@032026"

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    # End any leftover active run from the training pipeline
    if mlflow.active_run() is not None:
        logger.info("Ending previous active MLflow run before reduce_size.")
        mlflow.end_run()

    for name, approach in approaches.items():
        if not (hasattr(approach, "model") and approach.model is not None):
            logger.warning(
                f"Approach '{name}' has no trained model, skipping reduce_size."
            )
            continue

        model_name_full = cfg["model"][name].get("model_name", "")
        model_cls = model_name_full.split(".")[-1]
        logger.info(f"Applying reduce_size on '{name}' (class={model_cls})")

        resizer = BaseReduceSize(reduce_cfg)
        reduced = resizer.apply_reduce_size(
            model_name_cls=model_cls, darts_model=approach.model
        )

        logger.info(f"Evaluating reduced model for '{name}'...")
        # Evaluate reduced model
        from tsfb.base.schema.backend_config import ParallelBackendConfig
        from tsfb.base.utils.parallel import ParallelBackend

        original_model = approach.model
        approach.model = reduced

        eval_cfg = {
            "strategy_args": cfg["evaluation"]["strategy_args"],
            "metrics": cfg["evaluation"]["metrics"],
        }
        series_list = cfg["evaluation"]["series_list"]

        parallel_config = ParallelBackendConfig(backend="sequential", n_workers=1)
        ParallelBackend().init(parallel_config)
        try:
            result = eval_model(series_list, eval_cfg, approach)
            batches = list(result.collect())
            df_eval = (
                pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
            )
        finally:
            ParallelBackend().close(force=True)

        approach.model = original_model  # Restore

        logger.info(
            f"Reduce size eval completed for '{name}'. "
            f"Original size: {reduced.config.get('original_size_mb', 'N/A'):.4f} MB, "
            f"Quantizable ratio: {reduced.config.get('quantizable_param_ratio', 'N/A')}"
        )

        # Log to MLflow in a SINGLE run
        with mlflow.start_run(run_name=f"{experiment_name}_{name}_reduced"):
            mlflow.set_tag("pipeline", "ReduceSizePipeline")
            mlflow.set_tag("approach_name", name)
            mlflow.set_tag("reduce_method", reduce_cfg.method)
            mlflow.log_param("reduce_method", reduce_cfg.method)
            mlflow.log_param("target_ratio", reduce_cfg.target_ratio)
            mlflow.log_metric(
                "original_size_mb", reduced.config.get("original_size_mb", 0)
            )
            mlflow.log_metric(
                "quantizable_param_ratio",
                reduced.config.get("quantizable_param_ratio", 0),
            )

            # Log metrics from the latest evaluation
            if not df_eval.empty:
                numeric_cols = [
                    c
                    for c in df_eval.columns
                    if pd.api.types.is_numeric_dtype(df_eval[c])
                ]
                for metric_name in numeric_cols:
                    val = df_eval[metric_name].mean()  # Average if multiple
                    mlflow.log_metric(f"{metric_name}", float(val))
                    logger.info(f"Logged metric {metric_name}: {val}")

            # Log each reduced model file as an artifact
            for artifact_path in reduced.config.get("paths", []):
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
                    logger.info(f"Logged artifact: {artifact_path}")
                else:
                    logger.warning(f"Reduced artifact not found: {artifact_path}")

    logger.info("UC8 reduce_size, evaluation and MLflow logging completed.")


def retrieve_and_serve_model(config_path: str):
    """
    Step 2 & 3: Find the run from MLflow, load the ONNX forecast model,
    and evaluate with dummy data.
    """
    logger.info("===================== [2] RETRIEVE MLFLOW RUN =====================")

    # Load config to get experiment name
    cfg = load_config(config_path)
    set_env_vars_from_config(cfg)
    cfg = inject_hardcoded_configs(cfg)

    experiment_name = cfg["mlflow"]["experiment_name"]
    model_name = list(cfg["model"].keys())[0]  # Get the first specified model

    # Set env variables before initializing Pipeline
    tracking_uri = cfg.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if cfg.get("mlflow", {}).get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
        if "env_vars" not in cfg or "MLFLOW_S3_ENDPOINT_URL" not in cfg["env_vars"]:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.218:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "admin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "Dgs@032026"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Search for the nested child run that matches the model name.
    # The model artifact is logged in the child run, not the parent pipeline_run.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_name = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        logger.warning(
            f"No nested run named '{model_name}' found, falling back to latest run."
        )
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

    if not runs:
        raise ValueError("No runs found for this experiment.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    logger.info(f"Found run '{latest_run.info.run_name}' with ID: {run_id}")

    logger.info(
        "===================== [3] LOAD ONNX FORECAST MODEL ====================="
    )

    # Load ONNX model directly from the nested run using runs:/<run_id>/<artifact_path>
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/{model_name}"
    logger.info(f"Loading ONNX model from: {model_uri}")

    onnx_model = mlflow.onnx.load_model(model_uri)

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "model.onnx"
        onnx.save(onnx_model, str(onnx_path))

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=ort.get_available_providers(),
        )

    for inp in sess.get_inputs():
        logger.info(f"ONNX input: name={inp.name}, shape={inp.shape}, dtype={inp.type}")

    logger.info(
        "===================== [4] SERVING EVALUATION (DUMMY DATA) ====================="
    )
    horizon = cfg["evaluation"]["strategy_args"]["horizon"]
    input_chunk_length = cfg["model"][model_name]["model_hyper_params"].get(
        "input_chunk_length", 24
    )
    try:
        run_root_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/", dst_path=td
        )
        yaml_files = list(Path(run_root_dir).glob("*.yaml"))
        if yaml_files:
            with open(yaml_files[0], "r") as f:
                old_cfg = yaml.safe_load(f)
            for k, v in old_cfg.get("model", {}).items():
                if (
                    isinstance(v, dict)
                    and "model_hyper_params" in v
                    and "input_chunk_length" in v["model_hyper_params"]
                ):
                    input_chunk_length = v["model_hyper_params"]["input_chunk_length"]
                    break
    except Exception as e:
        logger.warning(
            f"Could not load old config yaml to infer input_chunk_length: {e}"
        )

    # Get target columns from config
    target_columns = cfg.get("data_module_config", {}).get("target_columns", ["OT"])
    n_features = len(target_columns)

    # Generate dummy input data
    index = pd.date_range("2024-01-01", periods=input_chunk_length, freq="h")
    raw_data = pd.DataFrame(
        {col: np.random.rand(input_chunk_length) for col in target_columns},
        index=index,
    )

    logger.info("===================== [5] BUILD ONNX FEEDS =====================")

    x_past = raw_data.iloc[-input_chunk_length:].to_numpy()[
        None, :, :
    ]  # (1, lookback, features)

    feeds = {}
    for inp in sess.get_inputs():
        dtype = np_dtype_from_onnx(inp.type)

        if inp.name == "x_past":
            feeds[inp.name] = x_past.astype(dtype)
        else:
            # For other inputs (e.g. x_future), create array with expected shape
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            feeds[inp.name] = np.zeros(shape, dtype=dtype)

    logger.info(f"ONNX feeds ready: { {k: v.shape for k, v in feeds.items()} }")

    logger.info("===================== [6] RUN ONNX INFERENCE =====================")

    yhat = sess.run(None, feeds)[0]

    # Remove batch dim if exists
    if yhat.ndim > 0 and yhat.shape[0] == 1:
        yhat = yhat[0]

    logger.info(f"Forecast completed! Output shape: {yhat.shape}")
    logger.info(f"Forecast sample (first 5): {yhat.flatten()[:5].tolist()}")
    logger.info("Model Serving and Validation Example Complete!")
    return run_id


def retrieve_and_serve_model_by_run_id(config_path: str, run_id: str):
    """
    Load the ONNX forecast model from a specific MLflow run ID and evaluate.
    """
    logger.info(
        f"===================== [7] LOAD MODEL FROM RUN ID {run_id} ====================="
    )

    cfg = load_config(config_path)
    set_env_vars_from_config(cfg)
    cfg = inject_hardcoded_configs(cfg)

    experiment_name = cfg["mlflow"]["experiment_name"]
    model_name = list(cfg["model"].keys())[0]

    tracking_uri = cfg.get("mlflow", {}).get(
        "tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    if cfg.get("mlflow", {}).get("tracking_uri"):
        os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
        if "env_vars" not in cfg or "MLFLOW_S3_ENDPOINT_URL" not in cfg["env_vars"]:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.218:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "admin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "Dgs@032026"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load ONNX model directly from run_id
    model_uri = f"runs:/{run_id}/{model_name}"
    logger.info(f"Loading model from: {model_uri}")
    onnx_model = mlflow.onnx.load_model(model_uri)

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "model.onnx"
        onnx.save(onnx_model, str(onnx_path))

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=ort.get_available_providers(),
        )

    for inp in sess.get_inputs():
        logger.info(f"ONNX input: name={inp.name}, shape={inp.shape}, dtype={inp.type}")

    logger.info(
        "===================== [8] SERVING EVALUATION BY RUN ID ====================="
    )
    input_chunk_length = cfg["model"][model_name]["model_hyper_params"].get(
        "input_chunk_length", 24
    )
    try:
        run_root_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/", dst_path=td
        )
        yaml_files = list(Path(run_root_dir).glob("*.yaml"))
        if yaml_files:
            with open(yaml_files[0], "r") as f:
                old_cfg = yaml.safe_load(f)
            for k, v in old_cfg.get("model", {}).items():
                if (
                    isinstance(v, dict)
                    and "model_hyper_params" in v
                    and "input_chunk_length" in v["model_hyper_params"]
                ):
                    input_chunk_length = v["model_hyper_params"]["input_chunk_length"]
                    break
    except Exception as e:
        logger.warning(
            f"Could not load old config yaml to infer input_chunk_length: {e}"
        )

    target_columns = cfg.get("data_module_config", {}).get("target_columns", ["OT"])

    index = pd.date_range("2024-01-01", periods=input_chunk_length, freq="h")
    raw_data = pd.DataFrame(
        {col: np.random.rand(input_chunk_length) for col in target_columns},
        index=index,
    )

    x_past = raw_data.iloc[-input_chunk_length:].to_numpy()[None, :, :]

    feeds = {}
    for inp in sess.get_inputs():
        dtype = np_dtype_from_onnx(inp.type)

        if inp.name == "x_past":
            feeds[inp.name] = x_past.astype(dtype)
        else:
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            feeds[inp.name] = np.zeros(shape, dtype=dtype)

    logger.info(f"ONNX feeds ready: { {k: v.shape for k, v in feeds.items()} }")

    yhat = sess.run(None, feeds)[0]

    if yhat.ndim > 0 and yhat.shape[0] == 1:
        yhat = yhat[0]

    logger.info(f"Forecast by run_id completed! Output shape: {yhat.shape}")
    logger.info(f"Forecast by run_id sample: {yhat.flatten()[:5].tolist()}")


if __name__ == "__main__":
    config_path = str(Path(__file__).resolve().parent / "configs/duong_example.yaml")

    # Step 1: Run Training Pipeline & Log to MLflow
    run_training_and_logging(config_path)

    # Small sleep to ensure MLflow backend finishes writing
    time.sleep(2)

    # -----------------------------------------------------------------------------------------
    # Step 1B: Separate step to evaluate the Old Model on Test set from the same configuration
    old_run_id = "cbb36ccd388c425799e178dd8eebbc3d"
    if old_run_id and old_run_id != "YOUR_SPECIFIC_RUN_ID":
        try:
            evaluate_old_model_from_mlflow(config_path, old_run_id)
        except Exception as e:
            logger.warning(f"Failed to evaluate old model from MLflow: {e}")
    # -----------------------------------------------------------------------------------------

    # Step 2 & 3: Retrieve from MLflow and Predict
    # run_id = retrieve_and_serve_model(config_path)

    # Step 4: Load model by explicitly providing run_id
    # if run_id:
    #     retrieve_and_serve_model_by_run_id(config_path, run_id)
