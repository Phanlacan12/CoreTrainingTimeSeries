import logging
from unittest.mock import MagicMock, patch

import pytest

import tsfb.reconstruct as rc


@pytest.fixture
def logger():
    return logging.getLogger("test_logger")


@pytest.fixture
def dummy_factory():
    return lambda: {"ok": True}


def test_conf_from_best_ok():
    best = {"name": "m1", "model_params": {"a": 1}}
    conf = rc._conf_from_best(best)
    assert conf["name"] == "m1"
    assert conf["approach"] == "default"


def test_conf_from_best_empty():
    with pytest.raises(ValueError):
        rc._conf_from_best({})


@patch("tsfb.reconstruct.get_models")
def test_build_model(mock_get_models, dummy_factory):
    mock_get_models.return_value = [dummy_factory]
    best = {"name": "m1", "model_params": {}}
    model = rc.build_model(best)
    assert model == {"ok": True}
    mock_get_models.assert_called_once()


@patch("tsfb.reconstruct.get_models")
def test_build_approaches_default(mock_get_models, dummy_factory, logger):
    mock_get_models.return_value = [dummy_factory]
    cfg = {"model": {"m1": {"approach": "default"}}}
    result = rc.build_approaches(cfg, {}, logger)
    app = result["m1"]
    assert isinstance(app, rc.DefaultApproach)
    assert app.name == "m1"


@patch("tsfb.reconstruct.get_models")
def test_build_approaches_univariate(mock_get_models, dummy_factory, logger):
    mock_get_models.return_value = [dummy_factory]
    cfg = {"model": {"m2": {"approach": "univariate_per_series"}}}
    app = rc.build_approaches(cfg, {}, logger)["m2"]
    assert isinstance(app, rc.UnivariateToMultivariate)


def test_build_approaches_hybrid(logger):
    cfg = {"model": {"m3": {"approach": "hybrid"}}}
    strategy_args = {"data_module_config": {"x": 1}}
    app = rc.build_approaches(cfg, strategy_args, logger)["m3"]

    assert isinstance(app, rc.HybridApproach)
    assert hasattr(app, "config")
    assert app.config["approach"] == "hybrid"


def test_build_approaches_invalid(logger):
    cfg = {"model": {"m4": {"approach": "weird"}}}
    with pytest.raises(ValueError):
        rc.build_approaches(cfg, {}, logger)


@patch("tsfb.reconstruct.build_approaches")
def test_build_approach_default(mock_build, logger):
    dummy = MagicMock(spec=rc.DefaultApproach)
    mock_build.return_value = {"m1": dummy}
    best = {"name": "m1", "model_params": {"approach": "default"}}
    app = rc._build_approach(best, logger)
    assert app == dummy


@patch("tsfb.reconstruct.HybridApproach")
def test_build_approach_hybrid(mock_hybrid, logger):
    conf = {"name": "m1", "approach": "hybrid", "group": [{"series": ["a"]}]}
    best = {"name": "m1", "model_params": conf, "strategy_args": {}}
    rc._build_approach(best, logger)
    mock_hybrid.assert_called_once()


@patch("tsfb.reconstruct._build_approach")
def test_build_approach_and_model_default(mock_build, dummy_factory):
    app = rc.DefaultApproach(model_factory=dummy_factory)
    mock_build.return_value = app
    best = {"name": "m1", "model_params": {"approach": "default"}}
    approach, model = rc.build_approach_and_model(best)
    assert isinstance(approach, rc.DefaultApproach)
    assert model == {"ok": True}


@patch("tsfb.reconstruct.get_single_model")
@patch("tsfb.reconstruct._build_approach")
def test_build_approach_and_model_hybrid(mock_build, mock_single):
    mock_single.return_value = {"m": True}
    app = rc.HybridApproach(config={"group": [{"name": "g1", "model_name": "M"}]})
    mock_build.return_value = app
    best = {"name": "mh", "model_params": {"approach": "hybrid"}}
    _, models = rc.build_approach_and_model(best)
    assert "g1" in models
    mock_single.assert_called_once()
