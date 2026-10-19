from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tsfb.base.evaluation.strategy.forecasting import ForecastingStrategy
from tsfb.base.mlflow.approach_handle import (
    LoggedModelResult,
    log_multi_models_under_approach_models,
    log_one_model_best_effort,
)
from tsfb.base.mlflow.model_logging import PyFuncLogSpec
from tsfb.base.mlflow.onnx_handle import ONNXLogSpec, log_model_onnx_only_if_supported


class DummyApproach:
    def __init__(self, model=None, models=None, name=None):
        self.model = model
        self.models = models
        self.name = name


class DummyOnnxModel:
    def to_onnx(self, path, **kwargs):
        del kwargs
        with open(path, "wb") as f:
            f.write(b"onnx")


def _fake_model_info(
    artifact_path="model_artifact",
    model_uri="runs:/run-123/model_artifact",
    run_id="run-123",
    model_id="model-id-123",
    registered_model_version=7,
):
    return SimpleNamespace(
        artifact_path=artifact_path,
        model_uri=model_uri,
        run_id=run_id,
        model_id=model_id,
        registered_model_version=registered_model_version,
    )


def test_log_model_onnx_uses_flavor_api_without_manual_registration(monkeypatch):
    fake_info = _fake_model_info(
        artifact_path="onnx_model",
        model_uri="runs:/run-123/onnx_model",
        run_id="run-123",
        model_id="onnx-model-id",
        registered_model_version=5,
    )
    fake_log_model = MagicMock(return_value=fake_info)
    monkeypatch.setattr(
        "tsfb.base.mlflow.onnx_handle.MLFLOW_ONNX_MODULE",
        SimpleNamespace(log_model=fake_log_model),
    )
    monkeypatch.setattr(
        "tsfb.base.mlflow.onnx_handle.ONNX_MODULE",
        SimpleNamespace(load=lambda _: object()),
    )
    monkeypatch.setattr(
        "tsfb.base.mlflow.onnx_handle.mlflow.active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-123")),
    )
    mock_log_artifacts = MagicMock()
    mock_register_model = MagicMock()
    monkeypatch.setattr(
        "tsfb.base.mlflow.onnx_handle.mlflow.log_artifacts",
        mock_log_artifacts,
    )
    monkeypatch.setattr(
        "tsfb.base.mlflow.onnx_handle.mlflow.register_model",
        mock_register_model,
    )

    ok, model_info, export_info = log_model_onnx_only_if_supported(
        DummyOnnxModel(),
        ONNXLogSpec(
            artifact_path="onnx_model",
            registered_model_name="reg_onnx_model",
        ),
    )

    assert ok is True
    assert model_info is fake_info
    assert export_info is not None
    fake_log_model.assert_called_once()
    call_kwargs = fake_log_model.call_args.kwargs
    assert call_kwargs["artifact_path"] == "onnx_model"
    assert call_kwargs["registered_model_name"] == "reg_onnx_model"
    assert call_kwargs["tags"]["model_role"] == "forecast"
    assert call_kwargs["tags"]["artifact_kind"] == "onnx"
    assert call_kwargs["metadata"]["export_method"] == "to_onnx"
    mock_log_artifacts.assert_not_called()
    mock_register_model.assert_not_called()


@patch(
    "tsfb.base.mlflow.approach_handle.log_model_onnx_only_if_supported",
    return_value=(False, None, None),
)
def test_log_one_model_pyfunc_uses_flavor_api_and_maps_model_info(mock_try_onnx):
    fake_info = _fake_model_info(
        artifact_path="forecast_model",
        model_uri="runs:/run-123/forecast_model",
        run_id="run-123",
        model_id="pyfunc-model-id",
        registered_model_version=8,
    )

    with patch(
        "tsfb.base.mlflow.model_logging.mlflow.pyfunc.log_model",
        return_value=fake_info,
    ) as mock_log_model, patch(
        "tsfb.base.mlflow.approach_handle.mlflow.register_model"
    ) as mock_register_model:
        result = log_one_model_best_effort(
            model_obj={"fake": "model"},
            model_name="forecast_model",
            horizon=3,
            register_model=True,
        )

    assert result == LoggedModelResult(
        ok=True,
        kind="pyfunc",
        model_name="forecast_model",
        artifact_path="forecast_model",
        export_method=None,
        exporter_module=None,
        model_id="pyfunc-model-id",
        model_uri="runs:/run-123/forecast_model",
        run_id="run-123",
        registered_model_name="forecast_model",
        registered_model_version=8,
    )
    assert mock_try_onnx.call_count == 1
    assert mock_log_model.call_count == 1
    call_kwargs = mock_log_model.call_args.kwargs
    assert call_kwargs["artifact_path"] == "forecast_model"
    assert call_kwargs["registered_model_name"] == "forecast_model"
    assert call_kwargs["tags"]["model_role"] == "forecast"
    assert call_kwargs["metadata"]["python_model"] == "ModelBasePyFunc"
    mock_register_model.assert_not_called()


@patch(
    "tsfb.base.mlflow.approach_handle.log_model_onnx_only_if_supported",
    return_value=(False, None, None),
)
def test_log_one_model_skips_registry_when_disabled(mock_try_onnx):
    fake_info = _fake_model_info(
        artifact_path="forecast_model",
        model_uri="runs:/run-123/forecast_model",
        run_id="run-123",
        model_id="pyfunc-model-id",
        registered_model_version=None,
    )

    with patch(
        "tsfb.base.mlflow.model_logging.mlflow.pyfunc.log_model",
        return_value=fake_info,
    ) as mock_log_model:
        result = log_one_model_best_effort(
            model_obj={"fake": "model"},
            model_name="forecast_model",
            horizon=3,
            register_model=False,
        )

    assert result.registered_model_name is None
    assert result.registered_model_version is None
    assert mock_try_onnx.call_count == 1
    assert mock_log_model.call_args.kwargs["registered_model_name"] is None


@patch("tsfb.base.mlflow.approach_handle.log_one_model_best_effort")
def test_log_multiple_models_preserves_nested_runs_and_names(mock_log_one):
    mock_log_one.return_value = LoggedModelResult(
        ok=True,
        kind="pyfunc",
        model_name="unused",
    )

    approach = DummyApproach(
        models={
            "A": DummyApproach(model={"fake": "A"}),
            "B": (DummyApproach(model={"fake": "B"}), None),
        }
    )

    with patch(
        "tsfb.base.mlflow.approach_handle.mlflow.start_run",
        side_effect=lambda **kwargs: nullcontext(),
    ) as mock_start_run:
        results = log_multi_models_under_approach_models(
            approach=approach,
            approach_name="hybrid_model",
            horizon=3,
            experiment_name="ForecastingBenchmark",
            register_model=True,
        )

    assert len(results) == 2
    assert mock_start_run.call_count == 2
    run_names = [call.kwargs["run_name"] for call in mock_start_run.call_args_list]
    assert run_names == [
        "ForecastingBenchmark_hybrid_model_A",
        "ForecastingBenchmark_hybrid_model_B",
    ]


def test_preprocess_logging_uses_pyfunc_helper_with_registry_name():
    strategy = SimpleNamespace(
        _preprocess_logged=False,
        logger=MagicMock(),
    )
    preprocess = object()
    approach = DummyApproach(name="test hybrid")
    fake_result = LoggedModelResult(
        ok=True,
        kind="pyfunc",
        model_name="My_Experiment_test_hybrid_preprocess",
        artifact_path="My_Experiment_test_hybrid_preprocess",
        model_uri="runs:/run-123/My_Experiment_test_hybrid_preprocess",
        run_id="run-123",
        registered_model_name="My_Experiment_test_hybrid_preprocess",
        registered_model_version=3,
    )

    with patch(
        "tsfb.base.evaluation.strategy.forecasting.mlflow.active_run",
        return_value=SimpleNamespace(
            info=SimpleNamespace(run_id="run-123", experiment_id="exp-1")
        ),
    ), patch(
        "tsfb.base.evaluation.strategy.forecasting.mlflow.get_experiment",
        return_value=SimpleNamespace(name="My Experiment"),
    ), patch(
        "tsfb.base.evaluation.strategy.forecasting.log_pyfunc_model",
        return_value=fake_result,
    ) as mock_log_pyfunc_model:
        ForecastingStrategy._maybe_log_preprocess_model(strategy, preprocess, approach)

    assert strategy._preprocess_logged is True
    mock_log_pyfunc_model.assert_called_once()
    spec = mock_log_pyfunc_model.call_args.args[0]
    assert isinstance(spec, PyFuncLogSpec)
    assert spec.model_name == "My_Experiment_test_hybrid_preprocess"
    assert spec.registered_model_name == "My_Experiment_test_hybrid_preprocess"
    assert spec.register_model is True
    assert spec.model_role == "preprocess"
    assert spec.extra_metadata["python_model"] == "PreprocessPyFunc"
