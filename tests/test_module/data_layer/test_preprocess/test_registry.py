import pytest

from tsfb.base.data_module.preprocess_model import (
    ASTGCRNPreprocess,
    DefaultPreprocessData,
    StaticGroupedPreprocess,
)
from tsfb.base.data_module.preprocess_model.preprocess_registry import (
    build_preprocess_from_config,
)


def test_default_preprocess_build():
    cfg = {"training_data_aspect_config": {"type": "default", "params": {}}}
    pre = build_preprocess_from_config(cfg)
    assert isinstance(pre, DefaultPreprocessData)


def test_default_implied_preprocess_build():
    cfg = {"training_data_aspect_config": {"params": {}}}
    pre = build_preprocess_from_config(cfg)
    assert isinstance(pre, DefaultPreprocessData)


def test_static_grouped_preprocess_build():
    cfg = {
        "training_data_aspect_config": {
            "type": "static_grouped",
            "params": {
                "time_col": "ts",
                "group_col": "id",
                "target_columns": ["y"],
            },
        }
    }
    pre = build_preprocess_from_config(cfg)
    assert isinstance(pre, StaticGroupedPreprocess)


def test_astgcrn_preprocess_build():
    cfg = {
        "training_data_aspect_config": {
            "type": "astgcrn",
            "params": {
                "num_of_hours": 1,
                "points_per_hour": 12,
                "num_for_predict": 3,
            },
        }
    }
    pre = build_preprocess_from_config(cfg)
    assert isinstance(pre, ASTGCRNPreprocess)


def test_unknown_preprocess_type():
    cfg = {"training_data_aspect_config": {"type": "unknown", "params": {}}}
    with pytest.raises(ValueError):
        build_preprocess_from_config(cfg)
