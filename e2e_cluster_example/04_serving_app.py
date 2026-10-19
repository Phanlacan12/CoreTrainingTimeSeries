"""E2E Serving App for clustered time-series forecasting.

Models are loaded from the MLflow Model Registry using the names recorded by
03_train_models.py in data/cluster_model_map.json (current tsfb logging
mechanism):
  - forecast model:   mlflow.onnx.load_model("models:/<name>/<version>")
  - preprocess model: mlflow.pyfunc.load_model("models:/<name>_preprocess/<version>")
    then unwrap_python_model() to reach the fitted StaticGroupedPreprocess.

Flow:
  1. Startup: load registry models + cluster mapping
  2. Serve(series_id, steps):
     a. Read offline features → filter by series_id
     b. Scale features using fitted preprocess model
     c. Build ONNX feeds → run inference
     d. Inverse-transform → return forecast
"""
import os
import pickle
import sys
import warnings
from pathlib import Path

# Suppress noisy warnings from matplotlib/mlflow/pyparsing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURESTORE_ROOT = (
    Path(__file__).resolve().parents[4] / "x06/feature_store_v4/timeseries-featurestore"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FEATURESTORE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURESTORE_ROOT))

import json
import os
import tempfile

os.environ["FEAST_REPO_DIR"] = "build_fs_epidemic"

os.chdir(Path(__file__).parent.resolve())

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import yaml

# ================================================================== #
#  Utility functions                                                   #
# ================================================================== #


def np_dtype_from_onnx(onnx_type: str):
    """Convert ONNX tensor type string to NumPy dtype."""
    t = (onnx_type or "").lower()
    for key, dtype in {
        "double": np.float64,
        "float16": np.float16,
        "float": np.float32,
        "int64": np.int64,
        "int32": np.int32,
        "bool": np.bool_,
    }.items():
        if key in t:
            return dtype
    return np.float32


def col_to_array(col_data, dtype):
    """Extract numpy array from DataFrame column or raw array."""
    if hasattr(col_data, "fillna"):
        return col_data.fillna(0).values.astype(dtype)
    return np.nan_to_num(np.array(col_data), nan=0.0).astype(dtype)


# ================================================================== #
#  Step 1: Load all models from the MLflow Model Registry              #
# ================================================================== #


def load_onnx_session(model_uri: str) -> ort.InferenceSession:
    """Load a registered ONNX model into an InferenceSession.

    The tsfb pipeline logs forecast models with mlflow.onnx.log_model, so we
    load them back with mlflow.onnx.load_model via a `models:/` URI.
    """
    print(f"  Loading ONNX model: {model_uri}")
    onnx_model = mlflow.onnx.load_model(model_uri)

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "model.onnx"
        onnx.save(onnx_model, str(onnx_path))
        sess = ort.InferenceSession(
            str(onnx_path), providers=ort.get_available_providers()
        )

    for inp in sess.get_inputs():
        print(f"    input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
    return sess


def load_preprocess_model(model_uri: str):
    """Load a registered PreprocessPyFunc model and unwrap it.

    The tsfb pipeline logs the fitted preprocessor as a pyfunc model
    (PreprocessPyFunc). We unwrap it to reach the inner
    StaticGroupedPreprocess, which exposes entity_scalers / inverse_predict.
    """
    print(f"  Loading preprocess model: {model_uri}")
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    prep_obj = pyfunc_model.unwrap_python_model().preprocess
    print(f"    type: {type(prep_obj).__name__}")
    print(f"    scaler keys: {list(prep_obj.entity_scalers.keys())}")
    return prep_obj


def load_all_cluster_models(cluster_model_map: dict):
    """Load ONNX + preprocess models for all clusters from the Model Registry.

    cluster_model_map: dict[cluster_id] -> {
        "forecast":   {"registered_model_name", "version", "model_uri"},
        "preprocess": {"registered_model_name", "version", "model_uri"},
    } as written by 03_train_models.py.

    Returns: dict[cluster_id] -> (ort.InferenceSession, preprocess_obj or None)
    """
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    loaded = {}
    for cluster_id, entry in cluster_model_map.items():
        print(f"\nLoading models for {cluster_id}:")

        forecast_info = entry.get("forecast")
        if not forecast_info:
            print(f"  WARNING: no forecast model registered for {cluster_id}, skipping")
            continue

        try:
            sess = load_onnx_session(forecast_info["model_uri"])
        except Exception as e:
            # The pipeline falls back to a pyfunc (ModelBasePyFunc) when the
            # model does not support ONNX export; this ONNX serving example
            # only handles the ONNX flavor.
            print(
                f"  WARNING: could not load '{forecast_info['model_uri']}' as ONNX "
                f"({e}). Skipping {cluster_id}."
            )
            continue

        prep_obj = None
        preprocess_info = entry.get("preprocess")
        if preprocess_info:
            try:
                prep_obj = load_preprocess_model(preprocess_info["model_uri"])
            except Exception as e:
                print(f"  WARNING: could not load preprocess model: {e}")

        loaded[cluster_id] = (sess, prep_obj)

    return loaded


# ================================================================== #
#  Step 2: Fetch features from offline store                           #
# ================================================================== #
def fetch_features(
    series_ids: list,
    timestamps: list = None,
    time_range: list = None,
) -> pd.DataFrame:
    """Dynamically fetch features extending into future timestamps cleanly calculating lags.
    Uses the underlying `Pipeline` natively to process range arrays and shifts.
    """
    import pandas as pd
    from timeseries_featurestore.pipeline import Pipeline

    pipeline = Pipeline("configs/epidemic_fs_config.yaml")

    print(f"  Fetching online TS pipeline features directly...")
    result_df = pipeline.run(
        job_type="serving",
        feature_service="training_service",
        series_ids=series_ids,
        timestamps=timestamps,
        time_range=time_range,
    )

    return result_df


# ================================================================== #
#  Step 3: Prepare features for ONNX                                   #
# ================================================================== #


def get_series_int_id(series_id: str, cluster_id: str, cluster_df: pd.DataFrame) -> int:
    """Map series_id string to integer index (matching training label encoding)."""
    if cluster_df is None:
        return 0

    cluster_series = sorted(
        cluster_df[cluster_df["cluster_id"] == cluster_id].index.tolist()
    )
    try:
        return cluster_series.index(series_id)
    except ValueError:
        return 0


def prepare_features(
    feature_df: pd.DataFrame,
    prep_obj,
    series_id: str,
    data_cfg: dict,
) -> pd.DataFrame:
    """Prepare features for ONNX inference (NO scaling applied).

    The ONNX model was trained on raw data, so we pass raw features.
    We only need to:
      - set series_id to the integer mapped_id (matching training)
      - ensure timestamp column exists
      - select/reorder columns to match training layout
    """
    prep_df = feature_df.copy()

    time_col = data_cfg.get("time_col", "timestamp")
    group_col = data_cfg.get("group_col", "series_id")

    # Ensure timestamp column aligns with user config's designated time_col
    if "event_timestamp" in prep_df.columns and time_col != "event_timestamp":
        prep_df.rename(columns={"event_timestamp": time_col}, inplace=True)
    elif "timestamp" in prep_df.columns and time_col != "timestamp":
        prep_df.rename(columns={"timestamp": time_col}, inplace=True)

    # The preprocessor now correctly receives string keys during training
    prep_df[group_col] = series_id

    # If prep_obj is available, USE IT to scale features
    if prep_obj is not None:
        result = prep_obj.predict(prep_df)

        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, dict) and "data" in result:
            result = result["data"]
    else:
        result = prep_df

    return result


# ================================================================== #
#  Step 4: Build ONNX feeds                                           #
# ================================================================== #


def build_onnx_feeds(
    sess: ort.InferenceSession,
    feature_proc: pd.DataFrame,
    data_cfg: dict,
    input_chunk_length: int,
    output_chunk_length: int,
) -> dict:
    """Build ONNX input feeds from preprocessed features."""
    target_cols = data_cfg.get("target_columns", [])
    cov = data_cfg.get("covariate", {})
    past_cols = cov.get("past", []) or []
    future_cols = cov.get("future", []) or []
    static_cols = cov.get("static", []) or []

    x_past_cols = target_cols + past_cols + future_cols

    proc_keys = (
        list(feature_proc.columns)
        if isinstance(feature_proc, pd.DataFrame)
        else list(feature_proc.keys())
        if hasattr(feature_proc, "keys")
        else []
    )

    feeds = {}
    for inp in sess.get_inputs():
        dtype = np_dtype_from_onnx(inp.type)
        shape = [s if isinstance(s, int) else 1 for s in inp.shape]

        if inp.name == "x_past":
            arr = np.zeros(shape, dtype=dtype)
            for i, col in enumerate(x_past_cols):
                if i >= shape[2]:
                    break
                if col in proc_keys:
                    vals = col_to_array(feature_proc[col], dtype)
                    slice_vals = vals[:input_chunk_length]
                    n = min(len(slice_vals), shape[1])
                    arr[0, -n:, i] = slice_vals[-n:]
            feeds[inp.name] = arr

        elif inp.name == "x_future":
            arr = np.zeros(shape, dtype=dtype)
            for i, col in enumerate(future_cols):
                if i >= shape[2]:
                    break
                if col in proc_keys:
                    vals = col_to_array(feature_proc[col], dtype)
                    slice_vals = vals[
                        input_chunk_length : input_chunk_length + output_chunk_length
                    ]
                    n = min(len(slice_vals), shape[1])
                    arr[0, -n:, i] = slice_vals[-n:]
            feeds[inp.name] = arr

        elif inp.name == "x_static":
            arr = np.zeros(shape, dtype=dtype)
            dim = shape[-1] if len(shape) > 0 else len(static_cols)
            for i, col in enumerate(static_cols):
                if i < dim and col in proc_keys:
                    vals = col_to_array(feature_proc[col], dtype)
                    val = float(vals[-1]) if len(vals) > 0 else 0.0
                    if len(shape) == 3:
                        arr[:, :, i] = val
                    else:
                        arr[:, i] = val
            feeds[inp.name] = arr

        else:
            feeds[inp.name] = np.zeros(shape, dtype=dtype)

    # if verbose:
    #     for k, v in feeds.items():
    #         print(f"  {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}")

    return feeds


# ================================================================== #
#  Step 5: Run inference                                                #
# ================================================================== #


def run_inference(sess: ort.InferenceSession, feeds: dict, steps: int) -> np.ndarray:
    """Run ONNX inference and return forecast array."""
    yhat = sess.run(None, feeds)[0]
    # if verbose:
    #     print(f"  Raw output shape: {yhat.shape}")
    return yhat.flatten()[:steps]


def inverse_transform(
    forecast: np.ndarray,
    prep_obj,
    series_id: str,
    target_cols: list,
) -> np.ndarray:
    """Inverse-transform scaled forecast back to original scale."""
    if prep_obj is None or not hasattr(prep_obj, "inverse_predict"):
        return forecast

    col_name = target_cols[0] if target_cols else "cases"
    df_pred = pd.DataFrame({col_name: forecast})

    inv_df = prep_obj.inverse_predict(df_pred, series_name=series_id)

    if col_name in inv_df.columns:
        return inv_df[col_name].to_numpy()
    return forecast


# ================================================================== #
#  Main serving class                                                  #
# ================================================================== #


class E2EServingApp:
    def __init__(self):
        try:
            self.cluster_df = pd.read_csv("data/cluster_mapping.csv").set_index(
                "series_id"
            )
            self.has_mapping = True
        except FileNotFoundError:
            print(
                "WARNING: data/cluster_mapping.csv not found. Operating in single-model 'default_cluster' mode."
            )
            self.cluster_df = None
            self.has_mapping = False

        with open("data/cluster_model_map.json", "r") as f:
            self.cluster_model_map = json.load(f)

        # Load FS transformer pipeline (for reference, but NOT applied to
        # offline_features.parquet which is already transformed)
        self._transformer_pipeline = None
        tp = Path("configs/.fs_config/training_service_transformers.pkl")
        if tp.exists():
            with open(tp, "rb") as f:
                self._transformer_pipeline = pickle.load(f)
            print("Loaded FS transformer pipeline (for raw data only)")

        # Load all models into RAM from the MLflow Model Registry
        self.models = load_all_cluster_models(self.cluster_model_map)
        print(f"\nReady. Loaded models for: {list(self.models.keys())}")

    def serve(self, series_id: str, forecast_start_time: str = None):
        print(f"\n{'='*60}")
        print(
            f"Serving: series_id={series_id}, forecast_start_time={forecast_start_time}"
        )
        print(f"{'='*60}")

        if self.has_mapping:
            if series_id not in self.cluster_df.index:
                raise ValueError(f"Unknown series_id: {series_id}")
            cluster_id = self.cluster_df.loc[series_id, "cluster_id"]
        else:
            cluster_id = "default_cluster"

        print(f"Cluster: {cluster_id}")

        if cluster_id not in self.models:
            raise ValueError(f"No loaded model for {cluster_id}")

        sess, prep_obj = self.models[cluster_id]

        with open(f"configs/config_{cluster_id}.yaml", "r") as f:
            cluster_cfg = yaml.safe_load(f)

        model_key = list(cluster_cfg["model"].keys())[0]
        model_params = cluster_cfg["model"][model_key].get("model_hyper_params", {})
        input_chunk_length = model_params.get("input_chunk_length", 24)
        output_chunk_length = model_params.get("output_chunk_length", 1)
        data_cfg = cluster_cfg.get("data_module_config", {})
        freq = data_cfg.get("freq", "h")

        if self.has_mapping:
            cluster_series = sorted(
                self.cluster_df[
                    self.cluster_df["cluster_id"] == cluster_id
                ].index.tolist()
            )
        else:
            if prep_obj is not None and hasattr(prep_obj, "entity_scalers"):
                cluster_series = sorted(list(prep_obj.entity_scalers.keys()))
            else:
                cluster_series = [series_id]

        if forecast_start_time is None:
            # Default to current time, floored to the dataset frequency
            start_ts = pd.Timestamp.now().floor(freq)
            forecast_start_time = str(start_ts)
        else:
            start_ts = pd.to_datetime(forecast_start_time)

        range_start = start_ts - pd.Timedelta(input_chunk_length, unit=freq.lower())
        range_end = start_ts + pd.Timedelta(
            max(0, output_chunk_length - 1), unit=freq.lower()
        )

        # Step 2: Fetch features dynamically via Feature Store Pipeline for future
        print(
            f"\n[Step 2] Fetch features for {len(cluster_series)} series in {cluster_id}"
        )
        feature_df = fetch_features(
            series_ids=cluster_series, time_range=[str(range_start), str(range_end)]
        )

        print(feature_df)

        # Step 3-5: prepare, build feed and run inference
        print("\n[Step 3-5] Prepare, Feed, and Infer for each series in cluster")
        forecasts = {}

        for sid in cluster_series:
            sid_df = feature_df[feature_df["series_id"] == sid].copy()
            if sid_df.empty:
                print(f"  [WARNING] No data found for {sid}, skipping.")
                continue

            # mapped_id = get_series_int_id(sid, cluster_id, self.cluster_df)

            target_cols = data_cfg.get("target_columns", [])
            feature_proc = prepare_features(sid_df, prep_obj, sid, data_cfg)
            feeds = build_onnx_feeds(
                sess,
                feature_proc,
                data_cfg,
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
            )
            steps = min(output_chunk_length, len(sid_df) - input_chunk_length)
            steps = max(1, steps)  # Fallback
            yhat = run_inference(sess, feeds, steps)
            yhat = inverse_transform(yhat, prep_obj, sid, target_cols)

            forecasts[sid] = yhat.tolist()

        result = forecasts.get(series_id, [])
        print(
            f"\n=> Forecast for requested {series_id} ({len(result)} steps): {result}"
        )
        return result


# ================================================================== #
#  Entry point                                                         #
# ================================================================== #


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = E2EServingApp()

    test_series = "LOC_001_DIS_01"
    app.serve(test_series, forecast_start_time="2023-01-23 00:00:00")
