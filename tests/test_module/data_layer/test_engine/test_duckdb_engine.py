import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine, text

from tsfb.base.data_module.engine import DuckDBEngine
from tsfb.base.data_module.source import CsvSource, ParquetSource, SqlAlchemySource


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "ts": pd.date_range("2025-01-01", periods=3, freq="D"),
        }
    )


def test_load_csv(tmp_path, sample_dataframe):
    # Save CSV
    file_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(file_path, index=False)

    src = CsvSource(
        file_path=str(file_path), index_col="ts", parse_dates=True, read_csv_kwargs={}
    )

    engine = DuckDBEngine()
    df = engine.load_csv(src, opts={})

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "value"]
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.shape == (3, 2)


def test_load_parquet(tmp_path, sample_dataframe):
    # Save Parquet
    file_path = tmp_path / "test.parquet"
    table = pa.Table.from_pandas(sample_dataframe)
    pq.write_table(table, file_path)

    src = ParquetSource(
        file_path=str(file_path),
        index_col="ts",
        columns=["id", "value", "ts"],
        read_parquet_kwargs={},
    )

    engine = DuckDBEngine()
    df = engine.load_parquet(src, opts={})

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "value"]
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.shape == (3, 2)


def test_load_csv_file_not_found():
    src = CsvSource(
        file_path="non_existent.csv",
        index_col=None,
        parse_dates=False,
        read_csv_kwargs={},
    )
    engine = DuckDBEngine()
    with pytest.raises(Exception):
        engine.load_csv(src, opts={})


def test_load_parquet_file_not_found():
    src = ParquetSource(
        file_path="non_existent.parquet",
        index_col=None,
        columns=None,
        read_parquet_kwargs={},
    )
    engine = DuckDBEngine()
    with pytest.raises(Exception):
        engine.load_parquet(src, opts={})


@pytest.fixture
def sqlite_db(tmp_path):
    """Create sqlite db, return sqlite url."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE items (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO items VALUES (1, 'apple'), (2, 'banana')"))
    return str(db_file)


def test_load_sqlalchemy_with_table(sqlite_db):
    url = f"sqlite:///{sqlite_db}"

    src = SqlAlchemySource(
        sqlalchemy_url=url,
        table="items",
        query=None,
        index_col=None,
        parse_dates=False,
        read_sql_kwargs={},
    )

    engine = DuckDBEngine()
    df = engine.load_sqlalchemy(src, opts={})

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "name"]
    assert df.shape == (2, 2)
    assert df["name"].tolist() == ["apple", "banana"]


def test_load_sqlalchemy_with_query(sqlite_db):
    url = f"sqlite:///{sqlite_db}"

    src = SqlAlchemySource(
        sqlalchemy_url=url,
        table=None,
        query="SELECT id FROM items WHERE id = 1",
        index_col=None,
        parse_dates=False,
        read_sql_kwargs={},
    )

    engine = DuckDBEngine()
    df = engine.load_sqlalchemy(src, opts={})

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id"]
    assert df.iloc[0]["id"] == 1


def test_load_sqlalchemy_without_table_or_query():
    url = "sqlite:///:memory:"
    src = SqlAlchemySource(
        sqlalchemy_url=url,
        table=None,
        query=None,
        index_col=None,
        parse_dates=False,
        read_sql_kwargs={},
    )

    engine = DuckDBEngine()
    with pytest.raises(ValueError):
        engine.load_sqlalchemy(src, opts={})
