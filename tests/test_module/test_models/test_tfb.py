import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.model_loader import get_single_model


def test_tfb_dlinear_forecast_fit_and_forecast():
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"y": np.sin(np.arange(20))}, index=index)
    model = get_single_model(
        "tfb.DLinear",
        {
            "input_chunk_length": 6,
            "output_chunk_length": 3,
            "batch_size": 4,
            "n_epochs": 1,
            "learning_rate": 1e-2,
            "freq": "D",
        },
    )
    model.forecast_fit(frame)
    pred = model.forecast(horizon=3, series=frame)
    assert pred.shape == (3, 1)


def test_tfb_xpatch_forecast_fit_and_forecast():
    index = pd.date_range("2024-01-01", periods=18, freq="D")
    frame = pd.DataFrame(
        {
            "y1": np.cos(np.arange(18)),
            "y2": np.sin(np.arange(18)),
        },
        index=index,
    )
    model = get_single_model(
        "tfb.xPatch",
        {
            "input_chunk_length": 6,
            "output_chunk_length": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "learning_rate": 1e-2,
            "freq": "D",
            "patch_len": 3,
            "stride": 2,
        },
    )
    model.forecast_fit(frame)
    pred = model.forecast(horizon=2, series=frame)
    assert pred.shape == (2, 2)


def test_tfb_dlinear_grouped_forecast_returns_grouped_frame():
    rows = []
    for gid in ("A", "B"):
        for offset, ts in enumerate(pd.date_range("2024-01-01", periods=16, freq="D")):
            rows.append({"timestamp": ts, "group_id": gid, "y": float(offset)})
    frame = pd.DataFrame(rows)
    model = get_single_model(
        "tfb.DLinear",
        {
            "input_chunk_length": 5,
            "output_chunk_length": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "learning_rate": 1e-2,
            "freq": "D",
        },
    )
    model.forecast_fit(
        frame,
        group_col="group_id",
        time_col="timestamp",
        target_cols=["y"],
    )
    pred = model.forecast(horizon=2, series=frame, covariates=None)
    assert list(pred.columns) == ["timestamp", "group_id", "y"]
    assert pred["group_id"].tolist() == ["A", "A", "B", "B"]


@pytest.mark.parametrize(
    ("model_name", "model_hyper_params"),
    [
        (
            "tfb.Transformer",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "d_model": 16,
                "d_ff": 32,
                "n_heads": 2,
                "e_layers": 1,
                "d_layers": 1,
            },
        ),
        (
            "tfb.TiDE",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 1,
                "d_layers": 1,
            },
        ),
        (
            "tfb.TimeKAN",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 1,
            },
        ),
        (
            "tfb.TimeMixer",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 1,
            },
        ),
        (
            "tfb.PDF",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "patch_len": 4,
                "stride": 2,
                "period": 4,
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 1,
                "n_heads": 2,
            },
        ),
        (
            "tfb.FITS",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "cut_freq": 2,
            },
        ),
        (
            "tfb.DUET",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 1,
                "n_heads": 2,
            },
        ),
        (
            "tfb.PatchMLP",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "d_model": 16,
                "d_ff": 32,
            },
        ),
        (
            "tfb.HDMixer",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "patch_len": 4,
                "stride": 2,
                "d_model": 16,
                "d_ff": 32,
                "n_heads": 2,
                "e_layers": 1,
            },
        ),
        (
            "tfb.Amplifier",
            {
                "input_chunk_length": 8,
                "output_chunk_length": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-2,
                "freq": "D",
                "hidden_size": 16,
            },
        ),
    ],
)
def test_tfb_phase_b_models_forecast_fit_and_forecast(
    model_name,
    model_hyper_params,
):
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    frame = pd.DataFrame({"y": np.sin(np.arange(24, dtype=float))}, index=index)
    model = get_single_model(model_name, model_hyper_params)
    model.forecast_fit(frame)
    pred = model.forecast(
        horizon=model_hyper_params["output_chunk_length"],
        series=frame,
    )
    assert pred.shape == (model_hyper_params["output_chunk_length"], 1)


def test_tfb_fedformer_supports_low_head_counts():
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    frame = pd.DataFrame({"y": np.sin(np.arange(24, dtype=float))}, index=index)
    model = get_single_model(
        "tfb.FEDformer",
        {
            "input_chunk_length": 8,
            "output_chunk_length": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "learning_rate": 1e-2,
            "freq": "D",
            "d_model": 16,
            "d_ff": 32,
            "n_heads": 2,
            "e_layers": 1,
            "d_layers": 1,
            "modes": 4,
        },
    )
    model.forecast_fit(frame)
    pred = model.forecast(horizon=2, series=frame)
    assert pred.shape == (2, 1)


def test_tfb_etsformer_no_device_index_mismatch():
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    frame = pd.DataFrame({"y": np.cos(np.arange(24, dtype=float))}, index=index)
    model = get_single_model(
        "tfb.ETSformer",
        {
            "input_chunk_length": 8,
            "output_chunk_length": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "learning_rate": 1e-2,
            "freq": "D",
            "d_model": 16,
            "d_ff": 32,
            "n_heads": 2,
            "e_layers": 1,
            "d_layers": 1,
            "top_k": 2,
            "device": "cpu",
        },
    )
    model.forecast_fit(frame)
    pred = model.forecast(horizon=2, series=frame)
    assert pred.shape == (2, 1)
