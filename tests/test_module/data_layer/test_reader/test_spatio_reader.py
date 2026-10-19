import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tsfb.base.data_module.read_data import SpatioReader


def _create_sample_csv(tmp_path: str) -> str:
    """Create a sample CSV file representing time–node–value data."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="H").tolist() * 2,
            "station": ["A"] * 4 + ["B"] * 4,
            "speed": [10, 20, 30, 40, 15, 25, 35, 45],
        }
    )
    file_path = os.path.join(tmp_path, "sample.csv")
    df.to_csv(file_path, index=False)
    return file_path


def _create_od_file(tmp_path: str, kind: str) -> str:
    """Create a sample OD (origin–destination) matrix in various formats."""
    arr = np.arange(4).reshape(2, 2).astype(np.float32)
    file_path = os.path.join(tmp_path, f"od_matrix.{kind}")

    if kind == "npy":
        np.save(file_path, arr)
    elif kind == "npz":
        np.savez(file_path, arr)
    elif kind == "csv":
        pd.DataFrame(arr).to_csv(file_path, header=False, index=False)
    elif kind == "parquet":
        pd.DataFrame(arr).to_parquet(file_path)
    else:
        raise ValueError("Unsupported kind")

    return file_path


def test_spatioreader_pivot_and_nodes(tmp_path):
    """Test pivoting data into wide format and extracting node names."""
    csv_path = _create_sample_csv(tmp_path)

    cfg = {
        "file_path": csv_path,
        "kind": "csv",
        "target_columns": ["timestamp", "station", "speed"],
        "index_column": "timestamp",
        "node_column": "station",
    }

    reader = SpatioReader(cfg)
    wide, _, _, _ = reader.load_data()

    # Verify that pivot worked and node names were captured
    assert isinstance(wide, pd.DataFrame)
    assert all(col in wide.columns for col in ["A_speed", "B_speed"])
    assert reader.nodes == ["A", "B"]
    assert wide.shape[0] == 4  # 4 time steps


@pytest.mark.parametrize("kind", ["npy", "npz", "csv", "parquet"])
def test_spatioreader_load_od_tensor(tmp_path, kind):
    """Test that OD tensors are loaded correctly for all supported formats."""
    csv_path = _create_sample_csv(tmp_path)
    od_path = _create_od_file(tmp_path, kind)

    cfg = {
        "file_path": csv_path,
        "kind": "csv",
        "target_columns": ["timestamp", "station", "speed"],
        "index_column": "timestamp",
        "node_column": "station",
        "od_path": od_path,
        "od_kind": kind,
    }

    reader = SpatioReader(cfg)
    reader.load_data()

    # Check that the OD tensor exists and has correct shape and dtype
    assert isinstance(reader.od_tensor, np.ndarray)
    assert reader.od_tensor.shape == (2, 2)
    assert reader.od_tensor.dtype == np.float32


def test_spatioreader_no_od_path(tmp_path):
    """Test that od_tensor is None when no OD path is provided."""
    csv_path = _create_sample_csv(tmp_path)

    cfg = {
        "file_path": csv_path,
        "kind": "csv",
        "target_columns": ["timestamp", "station", "speed"],
        "index_column": "timestamp",
        "node_column": "station",
    }

    reader = SpatioReader(cfg)
    reader.load_data()

    # OD tensor should not exist
    assert reader.od_tensor is None


def test_spatioreader_missing_columns(tmp_path):
    """Test that an error is raised if index or node columns are missing."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
    csv_path = os.path.join(tmp_path, "missing.csv")
    df.to_csv(csv_path, index=False)

    cfg = {
        "file_path": csv_path,
        "kind": "csv",
        "target_columns": ["time", "value"],
        "index_column": "timestamp",  # Does not exist
        "node_column": "station",  # Does not exist
    }

    reader = SpatioReader(cfg)

    # Expect a ValueError when required columns are missing
    with pytest.raises(ValueError, match="Input data must contain"):
        reader.load_data()
