from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.statistic_impl.negative_binomial import NegativeBinomialGLM


@pytest.fixture
def toy_nb_series():
    np.random.seed(42)
    n = 40
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]
    mu = 6.0 + 0.25 * np.arange(n)
    y = np.random.poisson(lam=mu, size=n)
    df = pd.DataFrame({"y": y}, index=pd.DatetimeIndex(dates))
    return df


def test_forecast_fit_and_forecast_with_ci(toy_nb_series):
    model = NegativeBinomialGLM(alpha=0.05, r=10.0)
    model.forecast_fit(toy_nb_series)

    horizon = 7
    result = model.forecast_with_ci(horizon)

    assert isinstance(result, dict)
    for key in ["mu_hat", "ci_lower", "ci_upper", "pred_lower", "pred_upper", "y_mode"]:
        assert key in result
        assert isinstance(result[key], np.ndarray)
        assert len(result[key]) == horizon
        assert np.all(np.isfinite(result[key]))

    assert np.all(result["ci_lower"] <= result["ci_upper"])
    assert np.all(result["pred_lower"] <= result["y_mode"])
    assert np.all(result["y_mode"] <= result["pred_upper"])


def test_forecast_vs_reference_series(toy_nb_series):
    model = NegativeBinomialGLM(alpha=0.1, r=8.0)
    model.forecast_fit(toy_nb_series)

    forecast = model.forecast(horizon=3, series=toy_nb_series)
    assert isinstance(forecast, np.ndarray)
    assert len(forecast) == 3
    assert (forecast >= 0).all()


def test_invalid_r_raises():
    with pytest.raises(ValueError):
        NegativeBinomialGLM(r=0.0)
    with pytest.raises(ValueError):
        NegativeBinomialGLM(r=-1.0)


def test_dispersion_affects_interval_width(toy_nb_series):
    m_small = NegativeBinomialGLM(alpha=0.05, r=3.0).forecast_fit(toy_nb_series)
    m_large = NegativeBinomialGLM(alpha=0.05, r=30.0).forecast_fit(toy_nb_series)

    h = 10
    out_small = m_small.forecast_with_ci(h)
    out_large = m_large.forecast_with_ci(h)

    width_small = out_small["pred_upper"] - out_small["pred_lower"]
    width_large = out_large["pred_upper"] - out_large["pred_lower"]

    assert np.nanmean(width_small) > np.nanmean(width_large)
