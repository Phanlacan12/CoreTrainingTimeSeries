import pandas as pd
import pytest

from tsfb.base.data_module.read_data import DataReader


def test_pandas_load_data(tmp_path):
    # Create a dummy CSV
    file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "target": [1, 2, 3, 4, 5],
            "past": [5, 4, 3, 2, 1],
            "future": [10, 20, 30, 40, 50],
            "static": [7, 7, 7, 7, 7],
        }
    )
    df.to_csv(file, index=False)
    config = {
        "file_path": str(file),
        "target_columns": ["target"],
        "covariate": {"past": ["past"], "future": ["future"], "static": ["static"]},
        "backend": "pandas",
        "index_col": None,
        "parse_dates": True,
    }
    reader = DataReader(config)
    tgt, past, future, static = reader.load_data()
    assert tgt.shape[0] == 5
    assert past.shape[1] == 1
    assert future.shape[1] == 1
    assert static.shape[1] == 1


def test_backend_spark_returns_engine(monkeypatch):
    config = {
        "file_path": "dummy.csv",
        "target_columns": ["target"],
        "backend": "spark",
    }
    reader = DataReader(config)
    from tsfb.base.data_module.engine import SparkEngine

    assert isinstance(reader.engine, SparkEngine)
