from __future__ import annotations

from collections.abc import Sized
from typing import List, Optional, Tuple, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tsfb.base.data_module.dataloader.base_deep_loader import BaseDeepDataLoader
from tsfb.base.data_module.preprocess_model.spatio_preprocess import SpatioSplits
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class SpatioTorchDataLoader(BaseDeepDataLoader):
    """
    Helper that builds PyTorch DataLoaders from the output of `SpatioPreprocess`.

    - `build_from_splits(...)` is used for training/validation/testing:
        + Input: `SpatioSplits = (x_splits, y_splits, nodes, od_tensor)`
        + Output: `train_loader, val_loader, test_loader`

    - `build_for_inference(...)` is used for inference:
        + Input: `SpatioInference = (X_full, Y_full, nodes, od_tensor)`
        + Output: `inference_loader`
    """

    def __init__(
        self,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        *,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            device=device,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        # spatio-specific metadata
        self.nodes: List[str] = []
        self.od_tensor: Optional[np.ndarray] = None

    def build_from_splits(
        self,
        splits: SpatioSplits,
    ) -> Tuple[
        DataLoader[Tuple[Tensor, ...]],
        DataLoader[Tuple[Tensor, ...]],
        DataLoader[Tuple[Tensor, ...]],
    ]:
        """
        Create train/val/test DataLoaders
        from the output of `SpatioPreprocess.fit(...)`.

        Args:
            splits: Tuple (x_splits, y_splits, nodes, od_tensor)

        Returns:
            (train_loader, val_loader, test_loader)
        """
        x_splits, y_splits, nodes, od_tensor = splits
        self.nodes = nodes
        self.od_tensor = od_tensor

        (
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
        ) = self._extract_xy_splits(x_splits, y_splits)

        self.train_loader = self._to_loader_supervised(train_x, train_y, shuffle=True)
        self.val_loader = self._to_loader_supervised(val_x, val_y, shuffle=False)
        self.test_loader = self._to_loader_supervised(test_x, test_y, shuffle=False)

        if self.train_loader is not None:
            sized_dataset = cast(Sized, self.train_loader.dataset)
            if len(sized_dataset) > 0:
                logger.info(
                    "SpatioTorchDataLoader: train sample shape = %s",
                    tuple(self.train_loader.dataset[0][0].shape),
                )
                logger.info(
                    "SpatioTorchDataLoader: val sample shape   = %s",
                    tuple(self.val_loader.dataset[0][0].shape),
                )
                logger.info(
                    "SpatioTorchDataLoader: test sample shape  = %s",
                    tuple(self.test_loader.dataset[0][0].shape),
                )
            else:
                logger.warning("SpatioTorchDataLoader: empty train dataset.")
        else:
            logger.warning("SpatioTorchDataLoader: train_loader is None.")

        return self.train_loader, self.val_loader, self.test_loader

    def get_adj_matrix(self) -> Optional[np.ndarray]:
        """Return the origin-destination / adjacency tensor if available."""
        return self.od_tensor
