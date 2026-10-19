from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

import torch

from tsfb.base.data_module.dataloader.astgcrn_loader import ASTGCRNDataLoader
from tsfb.base.data_module.dataloader.base_deep_loader import BaseDeepDataLoader
from tsfb.base.data_module.dataloader.spatio_loader import SpatioTorchDataLoader
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
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


PIPELINE_BUNDLES: Dict[str, Dict[str, Any]] = {
    # 1. DataReader + DefaultPreprocessData, no DataLoader
    "ts_default": {
        "reader_cls": DataReader,
        "preprocess_cls": DefaultPreprocessData,
        "dataloader_cls": None,
    },
    # 2). DataReader + StaticGroupedPreprocess, no DataLoader
    "static_grouped": {
        "reader_cls": DataReader,
        "preprocess_cls": StaticGroupedPreprocess,
        "dataloader_cls": None,
    },
    # 3. SpatioReader + SpatioPreprocess + SpatioTorchDataLoader
    "spatio_deep": {
        "reader_cls": SpatioReader,
        "preprocess_cls": SpatioPreprocess,
        "dataloader_cls": SpatioTorchDataLoader,
    },
    "astgcrn": {
        "reader_cls": SpatioReader,
        "preprocess_cls": ASTGCRNPreprocess,
        "dataloader_cls": ASTGCRNDataLoader,
    },
}


class DataPipelineFactory:
    """
    Factory that creates a (reader, preprocess, dataloader) bundle based on config.

    Expected config:
        config["data_pipeline"]["type"] in {
            "ts_default",
            "static_grouped",
            "spatio_deep",
        }

    Returns:
        (reader, preprocess, dataloader)
        - dataloader may be None if the chosen bundle does not define one.
    """

    DEFAULT_TYPE = "ts_default"

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
    ) -> Tuple[Any, PreprocessData, Optional[BaseDeepDataLoader]]:
        """
        Create (reader, preprocess, dataloader) from configuration.

        Args:
            config:
                - config["data_pipeline"]["type"] selects the pipeline bundle.
            device:
                - torch.device to use for the dataloader (if any).

        Returns:
            reader, preprocess, dataloader
        """
        pipeline_type = config.get("pipeline_type")

        if pipeline_type not in PIPELINE_BUNDLES:
            raise ValueError(
                f"Unsupported data pipeline type: {pipeline_type!r}. "
                f"Supported: {list(PIPELINE_BUNDLES.keys())}"
            )

        spec = PIPELINE_BUNDLES[pipeline_type]
        logger.info("Creating data pipeline type='%s'", pipeline_type)

        reader_cls: Type[Any] = spec["reader_cls"]
        preprocess_cls: Type[PreprocessData] = spec["preprocess_cls"]
        dataloader_cls: Optional[Type[BaseDeepDataLoader]] = spec.get("dataloader_cls")

        # 1. Reader instance
        reader = reader_cls(config)

        # 2. Preprocess instance
        preprocess = preprocess_cls(config)

        # 3. Dataloader instance (if defined)
        dataloader: Optional[BaseDeepDataLoader] = None
        if dataloader_cls is not None:
            batch_size = config.get("batch_size", 64)
            dataloader = dataloader_cls(
                batch_size=batch_size,
                device=device or torch.device("cpu"),
                drop_last=config.get("drop_last", False),
                num_workers=config.get("num_workers", 0),
            )

        return reader, preprocess, dataloader
