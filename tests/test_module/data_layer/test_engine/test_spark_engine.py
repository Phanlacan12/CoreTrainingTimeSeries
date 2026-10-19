import sqlite3

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tsfb.base.data_module.engine import SparkEngine
from tsfb.base.data_module.source import CsvSource, ParquetSource, SqlAlchemySource


@pytest.fixture(scope="module")
def spark_engine(tmp_path_factory):
    spark = (
        SparkSession.builder.appName("SparkEngineTest").master("local[1]")
        # .config("spark.jars", "./jar_file/sqlite-jdbc-3.45.2.0.jar")
        .getOrCreate()
    )
    yield SparkEngine(spark_session=spark)
    spark.stop()


def test_load_csv(spark_engine, tmp_path):
    df = pd.DataFrame(
        {
            "ts": ["2025-09-30 10:00:00", "2025-09-30 08:00:00", "2025-09-30 09:00:00"],
            "y": [3, 1, 2],
        }
    )
    csv_path = tmp_path / "demo.csv"
    df.to_csv(csv_path, index=False)

    src = CsvSource(file_path=str(csv_path), index_col="ts", parse_dates=True)
    df_loaded = spark_engine.load_csv(src, opts={})

    assert isinstance(df_loaded, pd.DataFrame)
    assert "y" in df_loaded.columns
    assert pd.api.types.is_datetime64_any_dtype(df_loaded.index)
    assert list(df_loaded["y"]) == [1, 2, 3]


def test_load_parquet(spark_engine, tmp_path):
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2025-10-01 10:00:00", "2025-10-01 08:00:00", "2025-10-01 09:00:00"]
            ),
            "y": [30, 10, 20],
        }
    )
    pq_path = tmp_path / "demo.parquet"
    df.to_parquet(
        pq_path,
        index=False,
        engine="pyarrow",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )

    src = ParquetSource(file_path=str(pq_path), index_col="ts", parse_dates=True)
    df_loaded = spark_engine.load_parquet(src, opts={})

    assert isinstance(df_loaded, pd.DataFrame)
    assert "y" in df_loaded.columns
    assert pd.api.types.is_datetime64_any_dtype(df_loaded.index)
    assert list(df_loaded["y"]) == [10, 20, 30]


@pytest.mark.parametrize(
    "url,expected_driver,expected_jdbc",
    [
        (
            "postgresql://user:pass@localhost:5432/dbname",
            "org.postgresql.Driver",
            "jdbc:postgresql://localhost:5432/dbname",
        ),
        (
            "mysql://user:pass@localhost:3306/dbname",
            "com.mysql.cj.jdbc.Driver",
            "jdbc:mysql://localhost:3306/dbname",
        ),
        (
            "mssql://user:pass@localhost:1433/dbname",
            "com.microsoft.sqlserver.jdbc.SQLServerDriver",
            "jdbc:sqlserver://localhost:1433;databaseName=dbname",
        ),
    ],
)
def test_to_jdbc_url(spark_engine, url, expected_driver, expected_jdbc):
    info = spark_engine._to_jdbc_url(url)
    assert info["url"] == expected_jdbc
    assert info["properties"]["driver"] == expected_driver
    assert info["properties"]["user"] == "user"
    assert info["properties"]["password"] == "pass"
