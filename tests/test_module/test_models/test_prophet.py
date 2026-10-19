import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.statistic_impl.prophet_statistic import ProphetStatistic


@pytest.fixture
def toy_prophet_series():
    idx = pd.date_range("2023-01-01", periods=60, freq="D")
    t = np.arange(len(idx), dtype=float)
    y = 10 + 0.1 * t + 0.5 * np.sin(2 * np.pi * t / 7)
    return pd.DataFrame({"y": y}, index=idx)


def assert_fc_dict(fc: dict, horizon: int) -> None:
    keys = ("mu_hat", "ci_lower", "ci_upper", "pred_lower", "pred_upper", "y_mode")
    assert set(keys) <= set(fc)
    for k in keys:
        assert isinstance(fc[k], np.ndarray) and len(fc[k]) == horizon
    # Prophet wrapper: mean CI is NaN; PI finite & ordered
    assert np.isnan(fc["ci_lower"]).all() and np.isnan(fc["ci_upper"]).all()
    assert np.isfinite(fc["mu_hat"]).all()
    assert np.isfinite(fc["pred_lower"]).all() and np.isfinite(fc["pred_upper"]).all()
    assert (fc["pred_lower"] <= fc["mu_hat"]).all() and (
        fc["mu_hat"] <= fc["pred_upper"]
    ).all()


def test_forecast_fit_and_forecast_with_ci(toy_prophet_series):
    model = ProphetStatistic(alpha=0.2)  # 80% PI
    model.forecast_fit(toy_prophet_series)
    out = model.forecast_with_ci(horizon=5)
    assert_fc_dict(out, 5)


def test_forecast_vs_reference_series(toy_prophet_series):
    model = ProphetStatistic(alpha=0.1).forecast_fit(toy_prophet_series)
    fc = model.forecast(horizon=3, series=toy_prophet_series)
    assert isinstance(fc, np.ndarray) and len(fc) == 3 and np.isfinite(fc).all()


def test_errors_when_not_fitted_or_bad_series(toy_prophet_series):
    model = ProphetStatistic()
    with pytest.raises(RuntimeError):
        model.forecast_with_ci(horizon=3)

    model.forecast_fit(toy_prophet_series)
    bad_df = pd.DataFrame({"y": [1, 2, 3]})  # no DatetimeIndex
    with pytest.raises(ValueError):
        model.forecast(horizon=2, series=bad_df)


def test_future_from_origin_infers_mode_delta():
    # Irregular gaps: 2d,1d,2d,1d -> mode is 1 day
    idx = pd.to_datetime(
        ["2023-01-01", "2023-01-03", "2023-01-04", "2023-01-06", "2023-01-07"]
    )
    series = pd.DataFrame({"y": [1, 2, 3, 4, 5]}, index=idx)

    model = ProphetStatistic()
    future_df = model._future_from_origin(series.index, horizon=3)  # noqa: SLF001

    assert list(future_df.columns) == ["ds"] and len(future_df) == 3
    assert future_df["ds"].iloc[0] == idx[-1] + pd.Timedelta(days=1)


def test_weekly_and_mode_configs(toy_prophet_series):
    model = ProphetStatistic(
        alpha=0.05, weekly_seasonality=True, seasonality_mode="multiplicative"
    )
    model.forecast_fit(toy_prophet_series)

    assert pytest.approx(model._prophet.interval_width, rel=1e-9) == 0.95
    assert "weekly" in model._prophet.seasonalities
    assert model._prophet.seasonality_mode == "multiplicative"

    assert_fc_dict(model.forecast_with_ci(horizon=3), 3)


def test_changepoint_and_holidays_configs(toy_prophet_series):
    holidays = pd.DataFrame(
        {
            "holiday": ["toy_holiday", "toy_holiday"],
            "ds": pd.to_datetime(["2023-01-10", "2023-02-14"]),
            "lower_window": [0, 0],
            "upper_window": [1, 1],
        }
    )
    model = ProphetStatistic(alpha=0.10, changepoint_prior_scale=0.5, holidays=holidays)
    model.forecast_fit(toy_prophet_series)

    assert pytest.approx(model._prophet.interval_width, rel=1e-9) == 0.90
    assert model._prophet.changepoint_prior_scale == 0.5
    assert model._prophet.holidays is not None
    assert "toy_holiday" in set(model._prophet.holidays["holiday"])

    assert_fc_dict(model.forecast_with_ci(horizon=3), 3)
