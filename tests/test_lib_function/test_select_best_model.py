import pandas as pd
import pytest
import yaml

from tsfb.run_and_select_best import select_best_model


def _write_yaml(path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


def test_select_best_auto_lower_rmse(tmp_path):
    """
    mode=auto with metric 'rmse' => should select the smallest value.
    Also verify parsing of strategy_args & model_params and config mapping.
    """
    cfg = {
        "models": [
            {"name": "model_a", "params": {"a": 1}},
            {"name": "model_b", "params": {"b": 2}},
        ]
    }
    yaml_path = tmp_path / "conf.yaml"
    _write_yaml(yaml_path, cfg)

    df = pd.DataFrame(
        {
            "approach_name": ["model_a", "model_b"],
            "rmse": [10.5, 9.2],
            "strategy_args": ['{"k": 1}', "{'k': 2}"],
            "model_params": ['{"alpha": 0.1}', "{'alpha': 0.2}"],
        }
    )

    out = select_best_model(
        df=df,
        config_path=str(yaml_path),
        metric="rmse",
        mode="auto",
    )

    assert out["name"] == "model_b"
    assert out["metric"].lower() == "rmse"
    assert out["score"] == pytest.approx(9.2)
    assert isinstance(out["strategy_args"], dict) and out["strategy_args"]["k"] == 2
    assert isinstance(out["model_params"], dict) and out["model_params"]["alpha"] == 0.2
    assert out["config"]["name"] == "model_b"
    assert out["config"]["params"]["b"] == 2


def test_select_best_higher_r2(tmp_path):
    """
    mode=higher with metric 'r2' => should select the largest value.
    """
    cfg = {
        "models": [
            {"name": "m1", "params": {"p": 1}},
            {"name": "m2", "params": {"p": 2}},
            {"name": "m3", "params": {"p": 3}},
        ]
    }
    yaml_path = tmp_path / "conf.yaml"
    _write_yaml(yaml_path, cfg)

    df = pd.DataFrame(
        {
            "approach_name": ["m1", "m2", "m3"],
            "r2": [0.71, 0.68, 0.74],
        }
    )

    out = select_best_model(
        df=df,
        config_path=str(yaml_path),
        metric="r2",
        mode="higher",
    )

    assert out["name"] == "m3"
    assert out["score"] == pytest.approx(0.74)
    assert out["config"]["params"]["p"] == 3


def test_select_best_parse_edge_json_and_none(tmp_path):
    """
    Check that _parse_json_field handles None/NaN and non-JSON strings:
    - Best row has None -> should return None
    - Non-JSON string -> should return the original string
    """
    cfg = {"models": [{"name": "a"}, {"name": "b"}]}
    yaml_path = tmp_path / "conf.yaml"
    _write_yaml(yaml_path, cfg)

    df = pd.DataFrame(
        {
            "approach_name": ["a", "b"],
            "rmse": [5.0, 4.0],  # b is better
            "strategy_args": [None, float("nan")],  # None and NaN => None
            "model_params": ["not-json", "not-json"],  # return raw string
        }
    )

    out = select_best_model(
        df=df,
        config_path=str(yaml_path),
        metric="rmse",
        mode="auto",
    )

    assert out["name"] == "b"
    assert out["strategy_args"] is None
    assert out["model_params"] == "not-json"


def test_select_best_raises_when_metric_missing(tmp_path):
    """
    If the metric column is missing, should raise ValueError.
    """
    cfg = {"models": [{"name": "a"}]}
    yaml_path = tmp_path / "conf.yaml"
    _write_yaml(yaml_path, cfg)

    df = pd.DataFrame({"approach_name": ["a"], "mae": [1.0]})

    with pytest.raises(ValueError, match=r"Metric 'rmse' not found"):
        select_best_model(
            df=df,
            config_path=str(yaml_path),
            metric="rmse",
            mode="auto",
        )
