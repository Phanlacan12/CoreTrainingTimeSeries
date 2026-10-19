from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsfb.base.models.model_loader import get_single_model


def test_tgcrn_forecast_fit_and_forecast():
    index = pd.date_range("2024-01-01", periods=16, freq="h")
    frame = pd.DataFrame(
        {
            "n0": np.linspace(1.0, 8.5, 16, dtype=np.float32),
            "n1": np.linspace(2.0, 9.5, 16, dtype=np.float32),
        },
        index=index,
    )
    model = get_single_model(
        "tgcrn.TGCRN",
        {
            "input_chunk_length": 4,
            "output_chunk_length": 2,
            "batch_size": 2,
            "n_epochs": 1,
            "learning_rate": 1e-2,
            "freq": "H",
            "device": "cpu",
            "points_per_hour": 1,
            "rnn_units": 8,
            "embed_dim": 4,
            "cheb_k": 2,
            "num_layers": 1,
            "graph_direction": "symm",
            "time_embedding": False,
            "time_station": False,
            "od_flag": False,
        },
    )
    loader = SimpleNamespace(nodes=["n0", "n1"], od_tensor=np.eye(2, dtype=np.float32))
    model.forecast_fit(frame, loader=loader)
    pred = model.forecast(horizon=2, series=frame)
    assert pred.shape == (2, 2)
