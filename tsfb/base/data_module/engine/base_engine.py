from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from tsfb.base.data_module.source import (
    CsvSource,
    ParquetSource,
    PrestoSource,
    SingleStoreSource,
    SqlAlchemySource,
)
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class Engine(ABC):
    """Abstract base class for data loading engines."""

    name: str

    @abstractmethod
    def load_csv(self, src: CsvSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a CSV source."""

    @abstractmethod
    def load_parquet(self, src: ParquetSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a Parquet source."""

    @abstractmethod
    def load_sqlalchemy(
        self, src: SqlAlchemySource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from a SQLAlchemy source."""

    @abstractmethod
    def load_singlestore(
        self, src: SingleStoreSource, opts: Dict[str, Any]
    ) -> pd.DataFrame:
        """Load data from a SingleStore source."""

    @abstractmethod
    def load_presto(self, src: PrestoSource, opts: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a Presto source."""
