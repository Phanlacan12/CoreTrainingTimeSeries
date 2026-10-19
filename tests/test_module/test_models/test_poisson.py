from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.statistic_impl.poisson_glm import PoissonGLM


@pytest.fixture
def toy_poisson_series():
    np.random.seed(42)
    n = 30
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]
    lam = 5 + 0.2 * np.arange(n)
    y = np.random.poisson(lam=lam)

    df = pd.DataFrame({"y": y}, index=pd.DatetimeIndex(dates))
    return df


def test_forecast_fit_and_forecast_with_ci(toy_poisson_series):
    model = PoissonGLM(alpha=0.05)
    model.forecast_fit(toy_poisson_series)

    horizon = 7
    result = model.forecast_with_ci(horizon)

    assert isinstance(result, dict)
    for key in ["mu_hat", "ci_lower", "ci_upper", "pred_lower", "pred_upper", "y_mode"]:
        assert key in result
        assert isinstance(result[key], np.ndarray)
        assert len(result[key]) == horizon
        assert not np.isnan(result[key]).any()

    assert np.all(result["pred_lower"] <= result["y_mode"])
    assert np.all(result["y_mode"] <= result["pred_upper"])


def test_forecast_vs_reference_series(toy_poisson_series):
    model = PoissonGLM()
    model.forecast_fit(toy_poisson_series)

    forecast = model.forecast(horizon=3, series=toy_poisson_series)
    assert isinstance(forecast, np.ndarray)
    assert len(forecast) == 3
    assert (forecast > 0).all()
