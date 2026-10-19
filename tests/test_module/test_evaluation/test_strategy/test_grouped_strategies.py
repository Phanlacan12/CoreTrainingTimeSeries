import json

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy.fixed_forecast import FixedForecast
from tsfb.base.evaluation.strategy.rolling_forecast import RollingForecast


def make_grouped_target(n_groups: int = 4, n_steps: int = 24) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2023-01-01", periods=n_steps, freq="D")
    for gid_idx in range(n_groups):
        gid = f"S{gid_idx + 1}"
        for step_idx, ts in enumerate(dates):
            rows.append(
                {
                    "timestamp": ts,
                    "series_id": gid,
                    "y": float(step_idx + gid_idx),
                }
            )
    return pd.DataFrame(rows)


def make_grouped_covariates(n_groups: int = 4, n_steps: int = 24):
    rows = []
    static_rows = []
    dates = pd.date_range("2023-01-01", periods=n_steps, freq="D")
    for gid_idx in range(n_groups):
        gid = f"S{gid_idx + 1}"
        static_rows.append({"series_id": gid, "static1": float(gid_idx + 1)})
        for step_idx, ts in enumerate(dates):
            rows.append(
                {
                    "timestamp": ts,
                    "series_id": gid,
                    "past_cov": float(step_idx),
                    "future_cov": float(step_idx + 10),
                }
            )
    df = pd.DataFrame(rows)
    return {
        "past_covariates": df[["timestamp", "series_id", "past_cov"]].copy(),
        "future_covariates": df[["timestamp", "series_id", "future_cov"]].copy(),
        "static_covariates": pd.DataFrame(static_rows),
    }


class MockGroupedLoader:
    group_col = "series_id"
    time_col = "timestamp"
    target_cols = ["y"]
    static_cols = ["static1"]

    def transform_frame(self, df):
        return df.copy()

    def inverse_transform_frame(self, df, **kwargs):
        return df.copy()


class MockGroupedApproach(ForecastingApproach):
    def __init__(self, output_shift: int = 0):
        super().__init__()
        self.output_shift = output_shift
        self.model = type(
            "ShiftAwareModel",
            (),
            {"config": {"output_chunk_shift": output_shift}},
        )()
        self.fit_calls = []
        self.forecast_calls = []

    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.fit_calls.append(
            {
                "train_data": train_data.copy(),
                "covariates": covariates,
                "kwargs": kwargs,
            }
        )
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        self.forecast_calls.append(
            {
                "horizon": horizon,
                "lookback_data": lookback_data.copy(),
                "covariates": covariates,
                "kwargs": kwargs,
            }
        )
        group_col = kwargs["group_col"]
        time_col = kwargs["time_col"]
        target_cols = kwargs["target_cols"]
        outputs = []
        for gid, group_df in lookback_data.groupby(group_col, sort=True):
            group_df = group_df.sort_values(time_col).reset_index(drop=True)
            pred_idx = pd.date_range(
                start=pd.to_datetime(group_df[time_col]).iloc[-1],
                periods=horizon + self.output_shift + 1,
                freq="D",
            )[self.output_shift + 1 :]
            pred_df = pd.DataFrame({time_col: pred_idx, group_col: gid})
            for target_col in target_cols:
                last_val = float(group_df[target_col].iloc[-1])
                pred_df[target_col] = (
                    last_val
                    + self.output_shift
                    + np.arange(1, horizon + 1, dtype=float)
                )
            outputs.append(pred_df)
        return pd.concat(outputs, ignore_index=True)

    def batch_forecast(self, *args, **kwargs):
        raise NotImplementedError


def test_fixed_forecast_grouped_executes_batch_per_dataset():
    strategy = FixedForecast(
        {
            "strategy_name": "fixed_forecast",
            "horizon": 4,
            "save_true_pred": False,
            "target_channel": None,
            "seed": 42,
            "deterministic": "full",
            "return_metrics_per_series": True,
            "data_module_config": {
                "pipeline_type": "static_grouped",
                "time_col": "timestamp",
                "group_col": "series_id",
                "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
            },
        },
        Evaluator(["mae", "mse", "rmse"]),
    )
    approach = MockGroupedApproach()
    results = strategy._execute(
        series=make_grouped_target(),
        meta_info=None,
        approach=approach,
        series_name="__default__",
        covariates=make_grouped_covariates(),
        loader=MockGroupedLoader(),
    )

    assert len(approach.fit_calls) == 1
    assert len(approach.forecast_calls) == 1
    per_series = json.loads(results[-1])
    assert set(per_series) == {"S1", "S2", "S3", "S4"}
    assert results[0] == 0.0


def test_fixed_forecast_grouped_respects_output_chunk_shift():
    strategy = FixedForecast(
        {
            "strategy_name": "fixed_forecast",
            "horizon": 2,
            "save_true_pred": False,
            "target_channel": None,
            "seed": 42,
            "deterministic": "full",
            "return_metrics_per_series": True,
            "data_module_config": {
                "pipeline_type": "static_grouped",
                "time_col": "timestamp",
                "group_col": "series_id",
                "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
            },
        },
        Evaluator(["mae", "mse", "rmse"]),
    )
    approach = MockGroupedApproach(output_shift=2)
    results = strategy._execute(
        series=make_grouped_target(n_groups=3, n_steps=24),
        meta_info=None,
        approach=approach,
        series_name="__default__",
        covariates=make_grouped_covariates(n_groups=3, n_steps=24),
        loader=MockGroupedLoader(),
    )

    assert len(approach.fit_calls) == 1
    assert len(approach.forecast_calls) == 1
    assert results[0] == 0.0


def test_rolling_forecast_grouped_executes_batch_windows():
    strategy = RollingForecast(
        {
            "strategy_name": "rolling_forecast",
            "horizon": 3,
            "stride": 2,
            "num_rollings": 3,
            "save_true_pred": False,
            "target_channel": None,
            "seed": 42,
            "deterministic": "full",
            "return_metrics_per_series": True,
            "data_module_config": {
                "pipeline_type": "static_grouped",
                "time_col": "timestamp",
                "group_col": "series_id",
                "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
            },
        },
        Evaluator(["mae", "mse", "rmse"]),
    )
    approach = MockGroupedApproach()
    results = strategy._execute(
        series=make_grouped_target(),
        meta_info=None,
        approach=approach,
        series_name="__default__",
        covariates=make_grouped_covariates(),
        loader=MockGroupedLoader(),
    )

    assert len(approach.fit_calls) == 1
    assert len(approach.forecast_calls) == 3
    per_series = json.loads(results[-1])
    assert set(per_series) == {"S1", "S2", "S3", "S4"}
    assert results[0] == 0.0


def test_rolling_forecast_grouped_respects_output_chunk_shift():
    strategy = RollingForecast(
        {
            "strategy_name": "rolling_forecast",
            "horizon": 2,
            "stride": 1,
            "num_rollings": 2,
            "save_true_pred": False,
            "target_channel": None,
            "seed": 42,
            "deterministic": "full",
            "return_metrics_per_series": True,
            "data_module_config": {
                "pipeline_type": "static_grouped",
                "time_col": "timestamp",
                "group_col": "series_id",
                "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
            },
        },
        Evaluator(["mae", "mse", "rmse"]),
    )
    approach = MockGroupedApproach(output_shift=2)
    results = strategy._execute(
        series=make_grouped_target(n_groups=3, n_steps=24),
        meta_info=None,
        approach=approach,
        series_name="__default__",
        covariates=make_grouped_covariates(n_groups=3, n_steps=24),
        loader=MockGroupedLoader(),
    )

    assert len(approach.fit_calls) == 1
    assert len(approach.forecast_calls) == 2
    assert results[0] == 0.0
