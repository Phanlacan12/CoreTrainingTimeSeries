from __future__ import annotations

from typing import Any, Dict
from urllib.parse import parse_qs, unquote_plus

import duckdb
import pandas as pd
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


class DuckDBEngine(Engine):
    """Engine for loading data using DuckDB."""

    name = "duckdb"

    def __init__(self, con=None):
        """Initialize DuckDBEngine."""
        self._con = con

    def _get_duck_con(self, opts):
        """Get or create DuckDB connection."""
        con = opts.get("duckdb_connection") or self._con
        if con is None:
            con = duckdb.connect()
        opts["duckdb_connection"] = con
        self._con = con
        return con

    def load_csv(self, src: CsvSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load CSV file to DataFrame using DuckDB."""
        con = self._get_duck_con(opts)
        try:
            logger.info("Loading CSV via DuckDB from %s", src.file_path)
            q = f"SELECT * FROM read_csv_auto('{src.file_path}')"
            pdf = con.sql(q).df()
            pdf = _process_index_and_dates(pdf, src)
            return pdf
        except FileNotFoundError as e:
            logger.error("CSV file not found: %s", e)
            raise
        except Exception as e:
            logger.exception(
                "Error loading CSV with DuckDB from %s: %s", src.file_path, e
            )
            raise

    def load_parquet(self, src: ParquetSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load Parquet file to DataFrame using DuckDB."""
        con = self._get_duck_con(opts)
        try:
            logger.info("Loading Parquet via DuckDB from %s", src.file_path)
            base = f"read_parquet('{src.file_path}')"
            if src.columns:
                cols = ", ".join(src.columns)
                q = f"SELECT {cols} FROM {base}"
            else:
                q = f"SELECT * FROM {base}"
            pdf = con.sql(q).df()
            pdf = _process_index_and_dates(pdf, src)
            return pdf
        except FileNotFoundError as e:
            logger.error("Parquet file not found: %s", e)
            raise
        except Exception as e:
            logger.exception(
                "Error loading Parquet with DuckDB from %s: %s", src.file_path, e
            )
            raise

    def load_sqlalchemy(
        self, src: SqlAlchemySource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from SQLAlchemy source using DuckDB."""
        con = self._get_duck_con(opts)
        alias = "ext_db"

        self._attach_from_url(con, src.sqlalchemy_url, alias=alias)
        if src.query:
            pdf = con.sql(src.query).df()
        elif src.table:
            fq = self._qualify_table(src.sqlalchemy_url, src.table, alias=alias)
            pdf = con.sql(f"SELECT * FROM {fq}").df()
        else:
            raise ValueError("SqlAlchemySource must have 'table' or 'query'.")

        pdf = _process_index_and_dates(pdf, src)
        return pdf

    def load_presto(self, src: PrestoSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a Presto source (Not Implemented)."""
        raise NotImplementedError("DuckDBEngine does not natively support Presto.")

    def load_singlestore(
        self, src: SingleStoreSource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from a SingleStore source (Not Implemented)."""
        raise NotImplementedError("DuckDBEngine does not natively support SingleStore.")

    def _backend_registry(self):
        """Return backend registry for DuckDB."""
        return {
            "mysql": {
                "duck_type": "MYSQL",
                "ext": "mysql_scanner",
                "schema_attr": "database",
                "build_attach_arg": lambda url: (
                    f"host={url.host or '127.0.0.1'} "
                    f"port={url.port or 3306} "
                    f"user={url.username or ''} "
                    f"password={url.password or ''} "
                    f"database={url.database or ''}"
                ),
            },
            "postgresql": {
                "duck_type": "POSTGRES",
                "ext": "postgres_scanner",
                "schema_attr": "database",
                "build_attach_arg": lambda url: (
                    f"host={url.host or '127.0.0.1'} "
                    f"port={url.port or 5432} "
                    f"user={url.username or ''} "
                    f"password={url.password or ''} "
                    f"database={url.database or ''}"
                ),
            },
            "postgres": {  # alias
                "duck_type": "POSTGRES",
                "ext": "postgres_scanner",
                "schema_attr": "database",
                "build_attach_arg": lambda url: (
                    f"host={url.host or '127.0.0.1'} "
                    f"port={url.port or 5432} "
                    f"user={url.username or ''} "
                    f"password={url.password or ''} "
                    f"database={url.database or ''}"
                ),
            },
            "sqlite": {
                "duck_type": "SQLITE",
                "ext": "sqlite_scanner",
                "schema_attr": None,
                "build_attach_arg": lambda url: url.database or ":memory:",
            },
            "mssql": {
                "duck_type": "ODBC",
                "ext": "odbc",
                "schema_attr": None,
                "build_attach_arg": self._extract_odbc_connect,
            },
        }

    def _qualify_table(
        self, sqlalchemy_url: str, table: str, alias: str = "ext_db"
    ) -> str:
        """Get fully qualified table name for DuckDB."""
        url = make_url(sqlalchemy_url)
        backend = self._normalize_backend_name(url.get_backend_name())
        if table.count(".") >= 1:
            return table
        if backend in ("mysql", "postgresql", "postgres"):
            db = url.database or ""
            return f"{alias}.{db}.{table}" if db else f"{alias}.{table}"
        if backend == "sqlite":
            return f"{alias}.{table}"
        return table

    def _normalize_backend_name(self, backend: str) -> str:
        """Normalize backend name string."""
        b = (backend or "").lower()
        if b.startswith("mysql"):
            return "mysql"
        if b in ("postgresql+psycopg2",):
            return "postgresql"
        if b.startswith("mssql"):
            return "mssql"
        return b

    def _extract_odbc_connect(self, url):
        """Extract ODBC connection string from URL."""
        if isinstance(url.query, dict):
            v = url.query.get("odbc_connect")
            if v:
                return unquote_plus(v[0] if isinstance(v, list) else v)

        try:
            qs = parse_qs(str(url.query))
            v = qs.get("odbc_connect")
            if v:
                return unquote_plus(v[0])
        except Exception:
            pass
        raise ValueError(
            "MSSQL must have URL like mssql+pyodbc:///?odbc_connect=ENCODED_DSN"
        )

    def _install_and_load(self, con, ext_name: str):
        """Install and load DuckDB extension."""
        con.execute(f"INSTALL {ext_name};")
        con.execute(f"LOAD {ext_name};")

    def _classify_backend(self, sqlalchemy_url: str):
        """Classify backend type and parse URL."""
        url = make_url(sqlalchemy_url)
        backend = self._normalize_backend_name(url.get_backend_name())
        reg = self._backend_registry()
        if backend not in reg:
            raise NotImplementedError(
                f"Backend '{backend}' is not supported for ATTACH via DuckDB"
            )
        return reg[backend], url

    def _attach_from_url(self, con, sqlalchemy_url: str, alias="ext_db"):
        """Attach external database to DuckDB connection."""
        spec, url = self._classify_backend(sqlalchemy_url)
        duck_type = spec["duck_type"]
        ext = spec["ext"]
        schema_attr = spec["schema_attr"]
        attach_arg = spec["build_attach_arg"](url)
        self._install_and_load(con, ext)
        try:
            con.execute(f"ATTACH '{attach_arg}' AS {alias} (TYPE {duck_type});")
        except Exception:
            pass

        backend = self._normalize_backend_name(url.get_backend_name())
        schema_name = None
        if schema_attr:
            schema_name = getattr(url, schema_attr, None)
        if backend == "sqlite" and not schema_name:
            schema_name = "main"

        if schema_name:
            con.execute(f"USE {alias}.{schema_name}")

        return alias
