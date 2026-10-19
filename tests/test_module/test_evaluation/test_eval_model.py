import pytest

from tsfb.base.evaluation.evaluate_model import eval_model
from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy.fixed_forecast import FixedForecast


class DummyTaskResult:
    def result(self):
        return []


class DummyApproach:
    name = "dummy"
    config = {}


@pytest.fixture
def sequential_schedule(monkeypatch):
    scheduled = []

    def _schedule(self, fn, args):
        scheduled.append(args[0])
        return DummyTaskResult()

    monkeypatch.setattr(
        "tsfb.base.evaluation.evaluate_model.ParallelBackend.schedule",
        _schedule,
    )
    return scheduled


def test_eval_model_static_grouped_schedules_single_dataset(sequential_schedule):
    evaluation_config = {
        "metrics": ["mae"],
        "strategy_args": {
            "strategy_name": "fixed_forecast",
            "seed": 42,
            "deterministic": "full",
            "horizon": 1,
            "save_true_pred": False,
            "target_channel": {"__default__": None},
            "data_module_config": {
                "pipeline_type": "static_grouped",
                "time_col": "timestamp",
                "group_col": "series_id",
                "target_columns": ["target"],
            },
        },
    }

    result = eval_model(["__default__"], evaluation_config, DummyApproach())

    assert sequential_schedule == ["__default__"]
    assert result.series_list == ["__default__"]
    assert len(result.result_list) == 1


def test_eval_model_static_grouped_rejects_multiple_aliases():
    evaluation_config = {
        "metrics": ["mae"],
        "strategy_args": {
            "strategy_name": "fixed_forecast",
            "seed": 42,
            "deterministic": "full",
            "horizon": 1,
            "save_true_pred": False,
            "target_channel": {"__default__": None},
            "data_module_config": {
                "pipeline_type": "static_grouped",
                "time_col": "timestamp",
                "group_col": "series_id",
                "target_columns": ["target"],
            },
        },
    }

    with pytest.raises(ValueError, match="exactly one dataset alias"):
        eval_model(["__default__", "shadow"], evaluation_config, DummyApproach())


def test_metrics_all_without_uncertainty_drops_interval_metrics():
    strategy = FixedForecast(
        {
            "strategy_name": "fixed_forecast",
            "seed": 42,
            "deterministic": "full",
            "horizon": 1,
            "save_true_pred": False,
            "target_channel": {"__default__": None},
            "data_module_config": {
                "target_columns": ["target"],
                "split_ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
            },
        },
        Evaluator("all"),
    )

    lowered = {name.lower() for name in strategy.evaluator.metric_names}
    assert "mae" in lowered
    assert "coverage" not in lowered
    assert "joint_coverage" not in lowered
    assert "mean_interval_width" not in lowered


def test_metrics_all_with_uncertainty_keeps_interval_metrics():
    strategy = FixedForecast(
        {
            "strategy_name": "fixed_forecast",
            "seed": 42,
            "deterministic": "full",
            "horizon": 1,
            "save_true_pred": False,
            "target_channel": {"__default__": None},
            "data_module_config": {
                "target_columns": ["target"],
                "split_ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
            },
            "uncertainty": {
                "method": "conformal",
                "alpha": 0.1,
                "calibration_source": "val",
                "bonferroni_correction": True,
                "finite_sample_correction": True,
            },
        },
        Evaluator("all"),
    )

    lowered = {name.lower() for name in strategy.evaluator.metric_names}
    assert "coverage" in lowered
    assert "joint_coverage" in lowered
    assert "mean_interval_width" in lowered
