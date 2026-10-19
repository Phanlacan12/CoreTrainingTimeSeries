"""Shared fixtures for foundation model tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def short_series() -> pd.DataFrame:
    """50-step univariate series — fast for unit tests."""
    np.random.seed(0)
    return pd.DataFrame({"value": np.cumsum(np.random.randn(50)) + 10.0})


@pytest.fixture
def long_series() -> pd.DataFrame:
    """200-step univariate series — enough context for foundation models."""
    np.random.seed(0)
    return pd.DataFrame({"value": np.cumsum(np.random.randn(200)) + 100.0})


@pytest.fixture
def multivariate_series() -> pd.DataFrame:
    np.random.seed(1)
    return pd.DataFrame(
        {
            "a": np.cumsum(np.random.randn(100)) + 10.0,
            "b": np.cumsum(np.random.randn(100)) + 20.0,
        }
    )


HORIZON = 12
