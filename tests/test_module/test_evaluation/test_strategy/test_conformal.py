import numpy as np
import pandas as pd
import pytest

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy.fixed_forecast import FixedForecast
from tsfb.base.evaluation.strategy.metric_utils import evaluate_metrics_with_group
from tsfb.base.evaluation.strategy.rolling_forecast import RollingForecast
from tsfb.base.uncertainty.conformal import (
    apply_symmetric_interval,
    calibrate_from_residual_windows,
)


class PerfectLinearApproach(ForecastingApproach):
    def forecast_fit(self, train_data, covariates=None, **kwargs):
        self.columns = list(train_data.columns)
        return self

    def forecast(self, horizon, lookback_data, covariates=None, **kwargs):
        group_col = kwargs.get("group_col")
        time_col = kwargs.get("time_col")
        target_cols = kwargs.get("target_cols")

        if (
            isinstance(lookback_data, pd.DataFrame)
            and group_col in lookback_data.columns
            and time_col in lookback_data.columns
        ):
            frames = []
            for gid, grp in lookback_data.groupby(group_col, sort=True):
                grp = grp.sort_values(time_col).reset_index(drop=True)
                cols = target_cols or [
                    c for c in grp.columns if c not in {group_col, time_col}
                ]
                pred = self._forecast_values(grp[cols], horizon)
                freq = pd.infer_freq(pd.to_datetime(grp[time_col])) or "D"
                next_idx = pd.date_range(
                    start=pd.to_datetime(grp[time_col].iloc[-1]),
                    periods=horizon + 1,
                    freq=freq,
                )[1:]
                pred[time_col] = next_idx
                pred[group_col] = gid
                frames.append(pred[[time_col, group_col] + cols])
            return pd.concat(frames, ignore_index=True)

        return self._forecast_values(lookback_data, horizon)

    def _forecast_values(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        out = {}
        for col in df.columns:
            values = df[col].to_numpy(dtype=float)
            step = values[-1] - values[-2] if len(values) >= 2 else 0.0
            out[col] = values[-1] + step * np.arange(1, horizon + 1, dtype=float)
        if isinstance(df.index, pd.DatetimeIndex):
            freq = pd.infer_freq(df.index) or "D"
            idx = pd.date_range(start=df.index[-1], periods=horizon + 1, freq=freq)[1:]
        else:
            idx = pd.RangeIndex(horizon)
        return pd.DataFrame(out, index=idx)

    def batch_forecast(self, *args, **kwargs):
        raise NotImplementedError


class IdentityGroupedLoader:
    group_col = "series_id"
    time_col = "timestamp"
    target_cols = ["target"]
    static_cols = []

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()

    def inverse_transform_frame(
        self, frame: pd.DataFrame, value_cols=None, series_name=None
    ) -> pd.DataFrame:
        return frame.copy()


def _metric_index(evaluator, name: str) -> int:
    mapping = {
        metric_name.lower(): idx
        for idx, metric_name in enumerate(evaluator.metric_names)
    }
    return mapping[name.lower()]


def test_conformal_quantiles_with_and_without_bonferroni():
    residuals = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0]],
            [[3.0], [4.0], [5.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )

    corrected = calibrate_from_residual_windows(
        residuals,
        horizon=3,
        alpha=0.2,
        bonferroni_correction=True,
        finite_sample_correction=True,
    )
    uncorrected = calibrate_from_residual_windows(
        residuals,
        horizon=3,
        alpha=0.2,
        bonferroni_correction=False,
        finite_sample_correction=False,
    )

    np.testing.assert_allclose(corrected.q_hat, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(uncorrected.q_hat, np.array([4.0, 5.0, 6.0]))
    assert corrected.alpha_effective < uncorrected.alpha_effective


def test_interval_metrics_with_grouped_windows():
    y_true = pd.DataFrame({"target": [1.0, 2.0, 10.0, 20.0]})
    y_pred = pd.DataFrame({"target": [1.1, 1.9, 11.0, 19.0]})
    lower, upper = apply_symmetric_interval(y_pred, [0.5, 0.5], target_cols=["target"])
    scores = evaluate_metrics_with_group(
        y_true=y_true,
        y_pred=y_pred,
        metric_names=["coverage", "joint_coverage", "mean_interval_width"],
        groups=np.array(["A", "A", "B", "B"]),
        interval_lower=lower,
        interval_upper=upper,
        window_ids=np.array(["A", "A", "B", "B"]),
    )

    assert scores["global"][0] == 50.0
    assert scores["global"][1] == 50.0
    assert scores["global"][2] == 1.0


def test_rolling_forecast_conformal_metrics():
    evaluator = Evaluator(["mae", "coverage", "joint_coverage", "mean_interval_width"])
    strategy = RollingForecast(
        {
            "strategy_name": "rolling_forecast",
            "horizon": 3,
            "stride": 1,
            "num_rollings": 2,
            "save_true_pred": True,
            "target_channel": None,
            "seed": 42,
            "deterministic": "full",
            "data_module_config": {
                "split_ratio": {"train": 0.5, "val": 0.25, "test": 0.25}
            },
            "uncertainty": {
                "method": "conformal",
                "alpha": 0.1,
                "calibration_source": "val",
                "calibration_stride": 1,
                "bonferroni_correction": True,
                "finite_sample_correction": True,
            },
        },
        evaluator,
    )
    series = pd.DataFrame(
        {"target": np.arange(20, dtype=float)},
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )

    results = strategy._eval_sample(
        series=series,
        meta_info=None,
        approach=PerfectLinearApproach(),
        series_name="demo",
        covariates=None,
        loader=None,
    )

    assert results[_metric_index(evaluator, "mae")] == pytest.approx(0.0, abs=1e-12)
    assert results[_metric_index(evaluator, "coverage")] == 100.0
    assert results[_metric_index(evaluator, "joint_coverage")] == 100.0
    assert results[_metric_index(evaluator, "mean_interval_width")] == 0.0
    assert isinstance(results[-3], str)


def test_fixed_forecast_grouped_conformal_alignment():
    evaluator = Evaluator(["mae", "coverage", "joint_coverage", "mean_interval_width"])
    strategy = FixedForecast(
        {
            "strategy_name": "fixed_forecast",
            "horizon": 2,
            "save_true_pred": True,
            "target_channel": None,
            "seed": 42,
            "deterministic": "full",
            "data_module_config": {
                "split_ratio": {"train": 0.5, "val": 0.25, "test": 0.25}
            },
            "uncertainty": {
                "method": "conformal",
                "alpha": 0.1,
                "calibration_source": "val",
                "calibration_stride": 1,
                "bonferroni_correction": True,
                "finite_sample_correction": True,
            },
        },
        evaluator,
    )
    timestamps = pd.date_range("2024-01-01", periods=12, freq="D")
    frames = []
    for offset, gid in enumerate(["A", "B"]):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "series_id": gid,
                    "target": np.arange(12, dtype=float) + offset * 100.0,
                }
            )
        )
    grouped_series = pd.concat(frames, ignore_index=True)

    results = strategy._execute_grouped(
        series=grouped_series,
        approach=PerfectLinearApproach(),
        series_name="demo_grouped",
        covariates=None,
        loader=IdentityGroupedLoader(),
    )

    assert results[_metric_index(evaluator, "mae")] == pytest.approx(0.0, abs=1e-12)
    assert results[_metric_index(evaluator, "coverage")] == 100.0
    assert results[_metric_index(evaluator, "joint_coverage")] == 100.0
    assert results[_metric_index(evaluator, "mean_interval_width")] == 0.0
