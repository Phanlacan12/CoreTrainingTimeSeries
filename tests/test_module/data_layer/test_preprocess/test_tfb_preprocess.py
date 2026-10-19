import numpy as np
import pandas as pd

from tsfb.base.data_module.preprocess_model.tfb_grouped_preprocess import (
    TFBGroupedPreprocess,
)
from tsfb.base.data_module.preprocess_model.tfb_preprocess import TFBPreprocess
from tsfb.base.mlflow.pyfunc_preprocess import PreprocessPyFunc


def test_tfb_preprocess_builds_train_and_inference_payloads():
    index = pd.date_range("2024-01-01", periods=18, freq="D")
    frame = pd.DataFrame({"y": np.arange(18, dtype=float)}, index=index)
    preprocess = TFBPreprocess(
        {
            "family": "seq2seq_with_time_mark",
            "seq_len": 4,
            "pred_len": 2,
            "label_len": 2,
            "batch_size": 2,
            "freq": "D",
            "target_columns": ["y"],
        }
    )

    payload = preprocess.fit(frame, train_ratio_in_tv=0.8)
    assert payload["mode"] == "train"
    assert payload["train_samples"]["x_enc"].shape[-2:] == (4, 1)
    assert payload["train_samples"]["y"].shape[-2:] == (2, 1)

    train_loader = preprocess.build_train_loader()
    batch = next(iter(train_loader))
    assert set(batch) == {"x_enc", "x_mark_enc", "x_dec", "x_mark_dec", "y"}

    inference = preprocess.predict(frame.iloc[-6:])
    assert inference["mode"] == "inference"
    assert inference["samples"]["x_enc"].shape == (1, 4, 1)
    assert len(inference["samples"]["prediction_index"]) == 1


def test_tfb_grouped_preprocess_keeps_group_boundaries():
    rows = []
    for gid in ("A", "B"):
        for idx, ts in enumerate(pd.date_range("2024-01-01", periods=10, freq="D")):
            rows.append({"group_id": gid, "timestamp": ts, "y": float(idx)})
    frame = pd.DataFrame(rows)
    preprocess = TFBGroupedPreprocess(
        {
            "family": "single_input_tensor",
            "seq_len": 3,
            "pred_len": 2,
            "label_len": 1,
            "batch_size": 4,
            "freq": "D",
            "group_col": "group_id",
            "time_col": "timestamp",
            "target_columns": ["y"],
        }
    )

    payload = preprocess.fit(frame, train_ratio_in_tv=0.7)
    assert payload["train_samples"]["x"].shape[-2:] == (3, 1)
    inference = preprocess.predict(frame)
    assert inference["samples"]["x"].shape[0] == 2
    assert inference["samples"]["group_id"] == ["A", "B"]


def test_preprocess_pyfunc_returns_structured_tfb_payload():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    frame = pd.DataFrame({"y": np.arange(8, dtype=float)}, index=index)
    preprocess = TFBPreprocess(
        {
            "family": "single_input_tensor",
            "seq_len": 4,
            "pred_len": 2,
            "label_len": 2,
            "batch_size": 2,
            "freq": "D",
            "target_columns": ["y"],
        }
    )
    preprocess.fit(frame, train_ratio_in_tv=1.0)

    pyfunc = PreprocessPyFunc(preprocess)
    result = pyfunc.predict(None, frame)
    assert result["mode"] == "inference"
    assert "samples" in result
