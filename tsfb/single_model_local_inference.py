# Local inference pipeline:
# pylint: disable=invalid-name
#   MLflow Preprocess (pyfunc)
#        -> select correct series
#        -> ONNX forecasting model
# ======================================================

import tempfile
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from numpy.typing import DTypeLike

from tsfb.base.utils.logging import get_logger, setup_logging

# ======================================================
# LOGGING
# ======================================================
setup_logging(lvl="INFO")
logger = get_logger("local_inference")


# ======================================================
# CONFIG
# ======================================================
PREPROCESS_NAME = "test_hybrid_preprocess"
FORECAST_MODEL_NAME = "test_hybrid_HUFL"
MODEL_VERSION = "latest"

HORIZON = 24
LOOKBACK = 24
FREQ = "h"

MODEL_TARGET_COLUMNS: List[str] = ["HUFL", "MUFL"]


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


logger.info("===================== [1] LOAD PREPROCESS MODEL =====================")

preprocess = mlflow.pyfunc.load_model(f"models:/{PREPROCESS_NAME}/{MODEL_VERSION}")

logger.info(
    "Loaded preprocess model",
    model_name=PREPROCESS_NAME,
    version=MODEL_VERSION,
)


logger.info(
    "===================== [2] LOAD ONNX FORECASTING MODEL ====================="
)

onnx_model = mlflow.onnx.load_model(f"models:/{FORECAST_MODEL_NAME}/{MODEL_VERSION}")

with tempfile.TemporaryDirectory() as td:
    onnx_path = Path(td) / "model.onnx"
    onnx.save(onnx_model, onnx_path)

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=ort.get_available_providers(),
    )

for inp in sess.get_inputs():
    logger.info(
        "ONNX input",
        name=inp.name,
        shape=inp.shape,
        dtype=inp.type,
    )


logger.info("===================== [3] BUILD RAW INPUT DATA =====================")

index = pd.date_range("2024-01-01", periods=72, freq=FREQ)

raw_target = pd.DataFrame(
    {
        "HUFL": np.random.rand(len(index)),
        "MUFL": np.random.rand(len(index)),
        "HULL": np.random.rand(len(index)),
        "LULL": np.random.rand(len(index)),
    },
    index=index,
)

raw_future = pd.DataFrame(
    {"OT": np.random.rand(len(index))},
    index=index,
)

raw_input = {
    "target": raw_target,
    "future_covariates": raw_future,
}

logger.info(
    "Raw input prepared",
    target_cols=list(raw_target.columns),
    future_cols=list(raw_future.columns),
    n_rows=len(index),
)


logger.info("===================== [4] PREPROCESS (FULL SERIES) =====================")

prep_out = preprocess.predict(raw_input)

target_proc: pd.DataFrame = prep_out["target_processed"]
future_proc: pd.DataFrame = prep_out["future_covariates_processed"]

logger.info(
    "Preprocess output",
    target_cols=list(target_proc.columns),
    future_cols=list(future_proc.columns),
)


logger.info(
    "===================== [5] SELECT MODEL-SPECIFIC SERIES ====================="
)

target_sel = target_proc[MODEL_TARGET_COLUMNS]

expected_targets = sess.get_inputs()[0].shape[2]
if target_sel.shape[1] != expected_targets:
    raise ValueError(
        f"ONNX model expects {expected_targets} target channels, "
        f"but got {target_sel.shape[1]} ({list(target_sel.columns)})"
    )

logger.info(
    "Selected target columns for model",
    selected_cols=MODEL_TARGET_COLUMNS,
)


logger.info("===================== [6] BUILD ONNX FEEDS =====================")

x_past = target_sel.iloc[-LOOKBACK:].to_numpy()[None, :, :]
x_future = future_proc.iloc[:HORIZON].to_numpy()[None, :, :]

feeds = {}

for inp in sess.get_inputs():
    dtype = np_dtype_from_onnx(inp.type)

    if inp.name == "x_past":
        feeds[inp.name] = x_past.astype(dtype)
    elif inp.name == "x_future":
        feeds[inp.name] = x_future.astype(dtype)
    else:
        raise RuntimeError(f"Unexpected ONNX input: {inp.name}")

logger.info(
    "ONNX feeds ready",
    x_past_shape=feeds["x_past"].shape,
    x_future_shape=feeds["x_future"].shape,
)


logger.info("===================== [7] RUN ONNX INFERENCE =====================")

yhat = sess.run(None, feeds)[0]

# remove batch dim if exists
if yhat.ndim == 3 and yhat.shape[0] == 1:
    yhat = yhat[0]

logger.info(
    "Forecast completed",
    output_shape=yhat.shape,
)

logger.info(
    "Forecast sample",
    values=yhat[:3].tolist(),
)
