from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from pyspark.sql import SparkSession
from sqlalchemy.engine.url import make_url

from tsfb.base.data_module.engine.base_engine import Engine
from tsfb.base.data_module.source import (
    CsvSource,
    ParquetSource,
    PrestoSource,
    SingleStoreSource,
    SqlAlchemySource,
)
from tsfb.base.utils.data_processing import _process_index_and_dates
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class SparkEngine(Engine):
    """Engine for loading data using Spark."""

    name = "spark"

    def __init__(
        self, spark_session: SparkSession | None = None, driver_jars: str | None = None
    ):
        """
        Initialize SparkEngine.

        Args:
            spark_session: existing SparkSession (optional)
            driver_jars: path(s) to JDBC driver jars (comma-separated or list)
            e.g. "/opt/jars/postgresql-42.7.3.jar,/opt/jars/mysql-connector-j-8.4.0.jar"
        """
        if spark_session is not None:
            self.spark = spark_session
        else:
            builder = SparkSession.builder.appName("SparkEngine")
            if driver_jars:
                if isinstance(driver_jars, (list, tuple)):
                    driver_jars = ",".join(driver_jars)
                builder = builder.config("spark.jars", driver_jars)
            self.spark = builder.getOrCreate()

    def _ensure_spark(self):
        """Ensure Spark session is available."""
        if self.spark is None:
            raise RuntimeError(
                "SparkEngine requires spark_session passed in engine_options['spark']."
            )

    def load_csv(self, src: CsvSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load CSV file to DataFrame using Spark."""
        self._ensure_spark()
        df = self.spark.read.csv(src.file_path, header=True, inferSchema=True)
        pdf = df.toPandas()
        pdf = _process_index_and_dates(pdf, src)
        return pdf

    def load_parquet(self, src: ParquetSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load Parquet file to DataFrame using Spark."""
        self._ensure_spark()
        df = self.spark.read.parquet(src.file_path)
        if src.columns:
            df = df.select(*src.columns)
        pdf = df.toPandas()
        pdf = _process_index_and_dates(pdf, src)
        return pdf

    def _to_jdbc_url(self, sqlalchemy_url: str) -> Dict[str, Any]:
        """Convert SQLAlchemy URL to JDBC URL and properties."""
        url = make_url(sqlalchemy_url)
        backend = url.get_backend_name().split("+")[0]

        if backend in ("postgresql", "postgres"):
            driver = "org.postgresql.Driver"
            jdbc_url = f"jdbc:postgresql://{url.host}:{url.port or 5432}/{url.database}"
        elif backend == "mysql":
            driver = "com.mysql.cj.jdbc.Driver"
            jdbc_url = f"jdbc:mysql://{url.host}:{url.port or 3306}/{url.database}"
        elif backend == "mssql":
            driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
            jdbc_url = (
                f"jdbc:sqlserver://{url.host}:{url.port or 1433};"
                f"databaseName={url.database}"
            )
        elif backend == "sqlite":
            driver = "org.sqlite.JDBC"
            jdbc_url = f"jdbc:sqlite:{url.database}"
        else:
            raise NotImplementedError(
                f"Backend {backend} is not supported for Spark JDBC"
            )

        props = {
            "user": url.username or "",
            "password": url.password or "",
            "driver": driver,
        }
        return {"url": jdbc_url, "properties": props}

    def load_by_jdbc(self, sqlalchemy_url: str, table: str = "", query: str = ""):
        """Load data from JDBC source, return Spark DataFrame."""
        self._ensure_spark()
        jdbc_info = self._to_jdbc_url(sqlalchemy_url)

        reader = self.spark.read.format("jdbc").option("url", jdbc_info["url"])
        for k, v in jdbc_info["properties"].items():
            reader = reader.option(k, v)

        if query:
            logger.info("Loading via Spark JDBC with query")
            reader = reader.option("query", query)
        elif table:
            logger.info("Loading via Spark JDBC with table=%s", table)
            reader = reader.option("dbtable", table)
        else:
            raise ValueError(
                "Either 'table' or 'query' must be provided for JDBC load."
            )

        return reader.load()

    def load_sqlalchemy(
        self, src: SqlAlchemySource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load SQLAlchemy source to DataFrame using Spark JDBC."""
        table = src.table if src.table is not None else ""
        query = src.query if src.query is not None else ""
        df = self.load_by_jdbc(src.sqlalchemy_url, table=table, query=query)
        pdf = df.toPandas()
        pdf = _process_index_and_dates(pdf, src)
        return pdf

    def load_presto(self, src: PrestoSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a Presto source (Not Implemented)."""
        raise NotImplementedError("SparkEngine does not natively support Presto yet.")

    def load_singlestore(
        self, src: SingleStoreSource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from a SingleStore source (Not Implemented)."""
        raise NotImplementedError(
            "SparkEngine does not natively support SingleStore yet."
        )
