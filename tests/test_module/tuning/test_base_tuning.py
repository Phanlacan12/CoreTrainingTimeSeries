import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfb.base.schema.backend_config import ParallelBackendConfig
from tsfb.base.utils.parallel import ParallelBackend
from tsfb.tuning.base_tuning import TRIAL_METRICS_USER_ATTR, BaseTuning
from tsfb.tuning.schema import ParameterType, TuningConfig, TuningParameter

TEST_CONFIG_PATH = REPO_ROOT / "tsfb_configs" / "static_test.yaml"


def test_base_tuning_initialization():
    """Test that BaseTuning can be initialized"""
    tuner = BaseTuning(study_name="test_study")
    assert tuner.study is not None
    assert tuner.direction == "minimize"


def test_tuning_config_metrics_eval_defaults_and_includes_objective():
    """metrics_eval defaults to the objective metric and always contains it."""
    tuning_config = TuningConfig(
        parameters=[TuningParameter(name="p1", type=ParameterType.INT, min=1, max=10)],
        metric="rmse",
    )
    assert tuning_config.metrics_eval == ["rmse"]

    tuning_config = TuningConfig(
        parameters=[TuningParameter(name="p1", type=ParameterType.INT, min=1, max=10)],
        metric="mae",
        metrics_eval=["MAPE", "mse", "mape"],
    )
    assert tuning_config.metrics_eval == ["MAPE", "mse", "mae"]


@pytest.fixture
def mock_config():
    if not TEST_CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {TEST_CONFIG_PATH}")

    with open(TEST_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Adjust config for faster test execution
    config["backend"]["type"] = "sequential"

    # Reduce complexity
    config["model"]["single_model"]["model_hyper_params"]["n_epochs"] = 1

    # Reduce rollings
    if "evaluation" in config and "strategy_args" in config["evaluation"]:
        config["evaluation"]["strategy_args"]["num_rollings"] = 2

    return config


def test_base_tuning_integration(mock_config, tmp_path, monkeypatch):
    """
    Test that BaseTuning runs an optimization loop using the static_test configuration.
    """
    # Ensure CWD is repo root so relative paths in config (like data path) work
    monkeypatch.chdir(REPO_ROOT)

    # Initialize ParallelBackend
    backend_config = ParallelBackendConfig(backend="sequential", n_workers=1, n_cpus=1)
    # Ensure any previous backend is closed
    try:
        ParallelBackend().close(force=True)
    except:
        pass

    ParallelBackend().init(backend_config)

    try:
        storage_url = None  # In-memory

        base_tuner = BaseTuning(
            study_name="test_tuning_integration",
            direction="minimize",
            storage=storage_url,
        )

        model_name = mock_config["model"]["single_model"]["model_name"]
        base_model_config = mock_config["model"]["single_model"]
        evaluation_config = mock_config["evaluation"]
        series_list = evaluation_config["series_list"]

        # Define a simple parameter sampler
        def param_sampler(trial):
            return {
                "input_chunk_length": trial.suggest_categorical(
                    "input_chunk_length", [12, 24]
                ),
            }

        # Run optimization
        study = base_tuner.optimize(
            model_name=model_name,
            base_model_config=base_model_config,
            evaluation_config=evaluation_config,
            series_list=series_list,
            param_sampler=param_sampler,
            n_trials=2,  # Very minimal
            metric_name="mse",  # Using 'mse' provided by MSE class
            n_jobs=1,
        )

        assert len(study.trials) == 2
        assert study.best_value is not None
        print(f"Best value: {study.best_value}")

        print(f"Best params: {study.best_params}")

    finally:
        ParallelBackend().close(force=True)


def test_tune_from_config(mock_config, tmp_path):
    """Test tuning directly from a model configuration entry."""
    storage_url = None  # In-memory
    base_tuner = BaseTuning(
        study_name="test_tune_direct",
        direction="minimize",
        storage=storage_url,
    )

    # Mimic what select_best_model returns
    model_entry = {
        "name": "darts.models.NLinearModel",
        "config": mock_config["model"]["single_model"],
        "score": 0.1,
    }

    evaluation_config = mock_config["evaluation"]
    series_list = evaluation_config["series_list"]

    def param_sampler(trial):
        return {"input_chunk_length": trial.suggest_categorical("icl", [12, 24])}

    # Initialize backend
    ParallelBackend().init(ParallelBackendConfig(backend="sequential", n_workers=1))
    try:
        study = base_tuner.tune_from_config(
            model_entry=model_entry,
            evaluation_config=evaluation_config,
            series_list=series_list,
            param_sampler=param_sampler,
            n_trials=1,
            metric_name="mse",
            n_jobs=1,
        )
        assert len(study.trials) == 1
    finally:
        ParallelBackend().close(force=True)


def test_tune_best_from_pipeline(mock_config, tmp_path, monkeypatch):
    """Test the full pipeline integration (with mocked PipelineExecutor)."""
    from unittest.mock import MagicMock

    mock_executor_cls = MagicMock()
    mock_executor_instance = mock_executor_cls.return_value
    mock_executor_instance.run_and_select_best.return_value = {
        "name": mock_config["model"]["single_model"]["model_name"],
        "config": mock_config["model"]["single_model"],
        "score": 0.05,
        "metric": "mse",
    }
    monkeypatch.setattr("tsfb.tuning.base_tuning.PipelineExecutor", mock_executor_cls)
    config_path = tmp_path / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)

    storage_url = f"sqlite:///{tmp_path}/test_optuna_pipeline_v2.db"
    base_tuner = BaseTuning(
        study_name="test_tune_pipeline",
        direction="minimize",
        storage=storage_url,
    )

    def param_sampler_provider(name):
        return lambda trial: {"input_chunk_length": 24}

    # Initialize backend
    ParallelBackend().init(ParallelBackendConfig(backend="sequential", n_workers=1))
    try:
        study = base_tuner.tune_best_from_pipeline(
            pipeline_config_path=str(config_path),
            param_sampler_provider=param_sampler_provider,
            results_path=str(tmp_path / "results.csv"),
            n_trials=1,
            metric_name="mse",
            n_jobs=1,
        )

        assert len(study.trials) == 1
        # Verify PipelineExecutor was called correctly
        mock_executor_cls.assert_called_with(
            config_path=str(config_path), results_path=str(tmp_path / "results.csv")
        )
        mock_executor_instance.run_and_select_best.assert_called_once()

    finally:
        ParallelBackend().close(force=True)


def test_tune_single_series(tmp_path, monkeypatch):
    """Test tune_single_series with multiple metrics and case-insensitive lookup."""
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    # Mock get_single_model
    mock_model_instance = MagicMock()
    # forecast should return numpy array
    mock_model_instance.forecast.return_value = np.array([1.0, 1.0])

    mock_get_model = MagicMock(return_value=mock_model_instance)
    monkeypatch.setattr("tsfb.tuning.base_tuning.get_single_model", mock_get_model)

    # Mock FixedForecast._evaluate_metrics
    # It returns multiple metrics to verify objective selection.
    monkeypatch.setattr(
        "tsfb.base.evaluation.strategy.fixed_forecast.FixedForecast._evaluate_metrics",
        MagicMock(return_value={"global": [0.2, 0.5], "per_series": {}}),
    )

    tuner = BaseTuning(study_name="test_single_series")

    tuning_config = TuningConfig(
        parameters=[TuningParameter(name="p1", type=ParameterType.INT, min=1, max=10)],
        n_trials=2,
        metric="rMsE",
        metrics_eval=["mae"],
    )

    y_true = np.array([1.0, 2.0])
    lookback_data = pd.DataFrame({"v": [1, 2]})

    study = tuner.tune_single_series(
        model_name="test_model",
        base_model_config={},
        tuning_config=tuning_config,
        y_true=y_true,
        lookback_data=lookback_data,
        catch_exceptions=False,
        log_to_mlflow=False,
    )

    assert len(study.trials) == 2
    assert study.best_value == pytest.approx(0.5)
    assert study.trials[0].user_attrs[TRIAL_METRICS_USER_ATTR] == {
        "MAE": 0.2,
        "RMSE": 0.5,
    }
    mock_get_model.assert_called()
    mock_model_instance.forecast_fit.assert_called()
    mock_model_instance.forecast.assert_called()


def test_tune_single_series_logs_all_metrics_to_mlflow(monkeypatch):
    """Top-K MLflow logging should include all metrics without re-fitting."""
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    mock_model_instance = MagicMock()
    mock_model_instance.forecast.return_value = np.array([1.0, 1.0])
    monkeypatch.setattr(
        "tsfb.tuning.base_tuning.get_single_model",
        MagicMock(return_value=mock_model_instance),
    )
    monkeypatch.setattr(
        "tsfb.base.evaluation.strategy.fixed_forecast.FixedForecast._evaluate_metrics",
        MagicMock(return_value={"global": [0.1, 0.2, 0.3, 0.4], "per_series": {}}),
    )

    mock_log_metric = MagicMock()
    monkeypatch.setattr("tsfb.tuning.base_tuning.mlflow.active_run", lambda: None)
    monkeypatch.setattr(
        "tsfb.tuning.base_tuning.mlflow.start_run",
        MagicMock(return_value=nullcontext()),
    )
    monkeypatch.setattr("tsfb.tuning.base_tuning.mlflow.log_metric", mock_log_metric)
    monkeypatch.setattr(
        "tsfb.tuning.base_tuning.mlflow.log_params",
        MagicMock(),
    )
    monkeypatch.setattr(
        "tsfb.tuning.base_tuning.mlflow.set_tag",
        MagicMock(),
    )

    tuner = BaseTuning(study_name="test_single_series_mlflow")
    tuning_config = TuningConfig(
        parameters=[TuningParameter(name="p1", type=ParameterType.INT, min=1, max=10)],
        n_trials=1,
        log_top_k_trial=1,
        metric="mae",
        metrics_eval=["mape", "mae", "mse", "rmse"],
    )

    study = tuner.tune_single_series(
        model_name="test_model",
        base_model_config={},
        tuning_config=tuning_config,
        y_true=np.array([1.0, 2.0]),
        lookback_data=pd.DataFrame({"v": [1.0, 2.0]}),
        catch_exceptions=False,
        log_to_mlflow=True,
    )

    assert study.best_value == pytest.approx(0.2)
    logged_metric_names = {call.args[0] for call in mock_log_metric.call_args_list}
    assert {"MAPE", "MAE", "MSE", "RMSE"}.issubset(logged_metric_names)
    assert mock_model_instance.forecast_fit.call_count == 1


def test_tune_single_series_integration(tmp_path):
    """Integration test for tune_single_series using PoissonGLM."""
    import numpy as np
    import pandas as pd

    from tsfb.tuning.base_tuning import BaseTuning
    from tsfb.tuning.schema import ParameterType, TuningConfig, TuningParameter

    n_points = 50
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    y = np.abs(np.sin(np.linspace(0, 10, n_points)) * 10).astype(int)
    df = pd.DataFrame({"value": y}, index=dates)

    horizon = 10
    lookback_data = df.iloc[:-horizon]
    y_true = df.iloc[-horizon:]["value"].values
    tuner = BaseTuning(study_name="test_integration")
    tuning_config = TuningConfig(
        parameters=[
            TuningParameter(
                name="alpha", type=ParameterType.FLOAT, min=0.01, max=0.2, step=0.1
            )
        ],
        n_trials=2,
        metric="rmse",
    )
    model_name = "statistic_impl.poisson_glm.PoissonGLM"
    base_model_config = {
        "model_name": model_name,
        "model_hyper_params": {},  # alpha injected by tuning
    }
    study = tuner.tune_single_series(
        model_name=model_name,
        base_model_config=base_model_config,
        tuning_config=tuning_config,
        y_true=y_true,
        lookback_data=lookback_data,
        catch_exceptions=False,
        log_to_mlflow=False,
    )

    assert len(study.trials) == 2
    assert study.best_value is not None
