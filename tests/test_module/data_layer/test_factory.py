import pytest
import torch

from tsfb.base.data_module.dataloader.astgcrn_loader import ASTGCRNDataLoader
from tsfb.base.data_module.dataloader.base_deep_loader import BaseDeepDataLoader
from tsfb.base.data_module.dataloader.spatio_loader import SpatioTorchDataLoader
from tsfb.base.data_module.factory import DataPipelineFactory
from tsfb.base.data_module.preprocess_model.astgcrn_preprocess import ASTGCRNPreprocess
from tsfb.base.data_module.preprocess_model.base_preprocess import (
    DefaultPreprocessData,
    PreprocessData,
)
from tsfb.base.data_module.preprocess_model.spatio_preprocess import SpatioPreprocess
from tsfb.base.data_module.preprocess_model.static_preprocess import (
    StaticGroupedPreprocess,
)
from tsfb.base.data_module.read_data import DataReader, SpatioReader


def _make_base_config() -> dict:
    """Create a minimal base config. `pipeline_type` will be set per test."""
    return {
        # Dataloader parameters
        "batch_size": 32,
        "drop_last": False,
        "num_workers": 0,
        # Minimal parameters for DataReader / SpatioReader / Preprocess
        "target_columns": ["y"],
        "covariate": {
            "past": [],
            "future": [],
            "static": [],
        },
        "kind": "csv",
        "file_path": "dummy.csv",
    }


def test_missing_pipeline_type_raises():
    """
    With the current implementation, if `pipeline_type` is missing,
    DataPipelineFactory.create() must raise a ValueError.
    """
    cfg = _make_base_config()

    with pytest.raises(ValueError) as exc_info:
        DataPipelineFactory.create(cfg)

    msg = str(exc_info.value)
    assert "Unsupported data pipeline type" in msg
    # At least assert that the message exposes the supported types
    assert "ts_default" in msg


def test_ts_default_pipeline_explicit_type():
    """Explicitly setting pipeline_type='ts_default' should create the default pipeline."""
    cfg = _make_base_config()
    cfg["pipeline_type"] = "ts_default"

    reader, preprocess, dataloader = DataPipelineFactory.create(cfg)

    assert isinstance(reader, DataReader)
    assert isinstance(preprocess, DefaultPreprocessData)
    # ts_default pipeline does not define a dataloader
    assert dataloader is None


def test_static_grouped_pipeline():
    """pipeline_type='static_grouped' should create DataReader + StaticGroupedPreprocess."""
    cfg = _make_base_config()
    cfg["pipeline_type"] = "static_grouped"

    reader, preprocess, dataloader = DataPipelineFactory.create(cfg)

    assert isinstance(reader, DataReader)
    assert isinstance(preprocess, StaticGroupedPreprocess)
    assert dataloader is None


def test_spatio_deep_pipeline_creates_correct_classes_and_params():
    """
    pipeline_type='spatio_deep' should create:
      - SpatioReader
      - SpatioPreprocess
      - SpatioTorchDataLoader with correct parameters.
    """
    cfg = _make_base_config()
    cfg["pipeline_type"] = "spatio_deep"

    # Set dataloader-specific parameters to verify that they are passed correctly
    cfg["batch_size"] = 16
    cfg["drop_last"] = True
    cfg["num_workers"] = 2

    device = torch.device("cpu")

    reader, preprocess, dataloader = DataPipelineFactory.create(cfg, device=device)

    # Reader & Preprocess should be the correct classes
    assert isinstance(reader, SpatioReader)
    assert isinstance(preprocess, SpatioPreprocess)

    # Dataloader must be created, of the correct class and with the correct params
    assert isinstance(dataloader, SpatioTorchDataLoader)
    assert isinstance(dataloader, BaseDeepDataLoader)

    assert dataloader.batch_size == cfg["batch_size"]
    assert dataloader.drop_last == cfg["drop_last"]
    assert dataloader.num_workers == cfg["num_workers"]
    assert dataloader.device == device


def test_astgcrn_pipeline_creates_correct_classes_and_params():
    """pipeline_type='astgcrn' should create the graph-aware TGCRN pipeline."""
    cfg = _make_base_config()
    cfg["pipeline_type"] = "astgcrn"
    cfg["batch_size"] = 8
    cfg["drop_last"] = False
    cfg["num_workers"] = 1

    device = torch.device("cpu")

    reader, preprocess, dataloader = DataPipelineFactory.create(cfg, device=device)

    assert isinstance(reader, SpatioReader)
    assert isinstance(preprocess, ASTGCRNPreprocess)
    assert isinstance(dataloader, ASTGCRNDataLoader)
    assert isinstance(dataloader, BaseDeepDataLoader)
    assert dataloader.batch_size == 8
    assert dataloader.num_workers == 1
    assert dataloader.device == device


def test_unsupported_pipeline_type_raises():
    """
    An unknown pipeline_type should raise ValueError and include the type name in the message.
    """
    cfg = _make_base_config()
    cfg["pipeline_type"] = "unknown_pipeline_type"

    with pytest.raises(ValueError) as exc_info:
        DataPipelineFactory.create(cfg)

    msg = str(exc_info.value)
    assert "Unsupported data pipeline type" in msg
    assert "unknown_pipeline_type" in msg
