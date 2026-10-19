from __future__ import annotations

import os
from typing import Any, Dict
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine

from tsfb.base.data_module.engine.base_engine import Engine
from tsfb.base.data_module.source import (
    CsvSource,
    ParquetSource,
    PrestoSource,
    SingleStoreSource,
    SqlAlchemySource,
)
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class PandasEngine(Engine):
    """Engine for loading data using pandas."""

    name = "pandas"

    def load_csv(self, src: CsvSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load CSV file to DataFrame."""
        try:
            logger.info("Loading CSV from %s", src.file_path)
            parse_dates = (
                [src.index_col]
                if src.index_col and src.parse_dates is True
                else src.parse_dates
            )
            df = pd.read_csv(
                src.file_path,
                index_col=src.index_col,
                parse_dates=parse_dates,
                **src.read_csv_kwargs,
            )
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except FileNotFoundError as e:
            logger.error("CSV file not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Error loading CSV from %s: %s", src.file_path, e)
            raise

    def load_parquet(self, src: ParquetSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load Parquet file to DataFrame."""
        try:
            logger.info("Loading Parquet from %s", src.file_path)
            df = pd.read_parquet(
                src.file_path, columns=src.columns, **src.read_parquet_kwargs
            )
            if src.index_col and src.index_col in df.columns:
                df = df.set_index(src.index_col)
            if src.index_col and not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    logger.warning(
                        "Could not parse index %s to datetime", src.index_col
                    )
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except FileNotFoundError as e:
            logger.error("Parquet file not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Error loading Parquet from %s: %s", src.file_path, e)
            raise

    def load_sqlalchemy(
        self, src: SqlAlchemySource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load SQLAlchemy source to DataFrame."""
        try:
            logger.info("Loading SQLAlchemy source: %s", src.sqlalchemy_url)
            parse_dates = (
                [src.index_col]
                if src.index_col and src.parse_dates is True
                else src.parse_dates
            )
            engine = create_engine(src.sqlalchemy_url)
            if src.query:
                df = pd.read_sql(
                    src.query,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            else:
                df = pd.read_sql_table(  # type: ignore[arg-type]
                    src.table,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            if src.index_col and src.index_col in df.columns:
                df = df.set_index(src.index_col)
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except Exception as e:
            logger.exception(
                "Error loading data from SQLAlchemy source (%s): %s",
                src.sqlalchemy_url,
                e,
            )
            raise

    def load_singlestore(
        self, src: SingleStoreSource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from SingleStore."""
        try:
            logger.info("Loading SingleStore source")
            parse_dates = (
                [src.index_col]
                if src.index_col and src.parse_dates is True
                else src.parse_dates
            )
            host = src.host or os.getenv("SINGLESTORE_HOST", "127.0.0.1")
            port = src.port or os.getenv("SINGLESTORE_PORT", "3306")
            user = src.user or os.getenv("SINGLESTORE_USER", "root")
            password = src.password or os.getenv("SINGLESTORE_PASS", "")
            database = src.database or os.getenv("SINGLESTORE_DB", "")
            password = quote_plus(password) if password else ""
            # Assuming pymysql is installed
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(url)

            if src.query:
                df = pd.read_sql(
                    src.query,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            elif src.table:
                df = pd.read_sql_table(  # type: ignore[arg-type]
                    src.table,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            else:
                raise ValueError(
                    "Either query or table must be provided for SingleStore"
                )

            if src.index_col and src.index_col in df.columns:
                df = df.set_index(src.index_col)
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except Exception as e:
            logger.exception("Error loading from SingleStore: %s", e)
            raise

    def load_presto(self, src: PrestoSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from Presto."""
        try:
            logger.info("Loading Presto source")
            parse_dates = (
                [src.index_col]
                if src.index_col and src.parse_dates is True
                else src.parse_dates
            )
            host = src.host or os.getenv("PRESTO_HOST", "localhost")
            port = src.port or os.getenv("PRESTO_PORT", "8080")
            user = src.user or os.getenv("PRESTO_USERNAME", "admin")
            password = src.password or os.getenv("PRESTO_PASSWORD", "")
            catalog = src.catalog or os.getenv("PRESTO_CATALOG", "hive")
            schema = src.schema or os.getenv("PRESTO_SCHEMA", "default")

            if password:
                url = f"presto://{user}:{password}@{host}:{port}/{catalog}/{schema}"
            else:
                url = f"presto://{user}@{host}:{port}/{catalog}/{schema}"

            engine = create_engine(url)

            if src.query:
                df = pd.read_sql(
                    src.query,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            elif src.table:
                df = pd.read_sql_table(  # type: ignore[arg-type]
                    src.table,
                    con=engine,
                    parse_dates=parse_dates,
                    **src.read_sql_kwargs,
                )
            else:
                raise ValueError("Either query or table must be provided for Presto")

            if src.index_col and src.index_col in df.columns:
                df = df.set_index(src.index_col)
            if src.index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            return df
        except Exception as e:
            logger.exception("Error loading from Presto: %s", e)
            raise
