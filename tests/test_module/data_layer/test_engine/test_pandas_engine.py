import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine

from tsfb.base.data_module.engine import PandasEngine
from tsfb.base.data_module.source import CsvSource, ParquetSource, SqlAlchemySource


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})


def test_pandas_load_csv_success(tmp_path, sample_dataframe):
    file_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(file_path, index=False)

    src = CsvSource(
        file_path=str(file_path), index_col=None, parse_dates=False, read_csv_kwargs={}
    )
    df = PandasEngine().load_csv(src, {})

    pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_dataframe)


def test_pandas_load_csv_file_not_found():
    src = CsvSource(
        file_path="not_exists.csv",
        index_col=None,
        parse_dates=False,
        read_csv_kwargs={},
    )
    with pytest.raises(FileNotFoundError):
        PandasEngine().load_csv(src, {})


def test_pandas_load_parquet_success(tmp_path, sample_dataframe):
    file_path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pandas(sample_dataframe), file_path)

    src = ParquetSource(
        file_path=str(file_path), index_col=None, columns=None, read_parquet_kwargs={}
    )
    df = PandasEngine().load_parquet(src, {})

    pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_dataframe)


def test_pandas_load_parquet_file_not_found():
    src = ParquetSource(
        file_path="not_exists.parquet",
        index_col=None,
        columns=None,
        read_parquet_kwargs={},
    )
    with pytest.raises(FileNotFoundError):
        PandasEngine().load_parquet(src, {})


def test_pandas_load_sqlalchemy_query(sample_dataframe, monkeypatch):
    engine_sql = create_engine("sqlite:///:memory:")
    sample_dataframe.to_sql("test_table", engine_sql, index=False)

    monkeypatch.setattr(
        "tsfb.base.data_module.engine.pandas_engine.create_engine",
        lambda url: engine_sql,
    )

    src = SqlAlchemySource(
        sqlalchemy_url="sqlite:///:memory:",
        table="test_table",
        query="SELECT * FROM test_table",
        index_col=None,
        parse_dates=False,
        read_sql_kwargs={},
    )
    df = PandasEngine().load_sqlalchemy(src, {})
    pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_dataframe)


def test_pandas_load_sqlalchemy_table(sample_dataframe, monkeypatch):
    engine_sql = create_engine("sqlite:///:memory:")
    sample_dataframe.to_sql("test_table", engine_sql, index=False)

    monkeypatch.setattr(
        "tsfb.base.data_module.engine.pandas_engine.create_engine",
        lambda url: engine_sql,
    )

    src = SqlAlchemySource(
        sqlalchemy_url="sqlite:///:memory:",
        table="test_table",
        query=None,
        index_col=None,
        parse_dates=False,
        read_sql_kwargs={},
    )
    df = PandasEngine().load_sqlalchemy(src, {})
    pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_dataframe)


def test_pandas_load_sqlalchemy_fail():
    src = SqlAlchemySource(
        sqlalchemy_url="invalid://",
        table="test_table",
        query=None,
        index_col=None,
        parse_dates=False,
        read_sql_kwargs={},
    )
    with pytest.raises(Exception):
        PandasEngine().load_sqlalchemy(src, {})
