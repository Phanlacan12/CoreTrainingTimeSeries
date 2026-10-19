# pylint: disable=duplicate-code
from __future__ import annotations

from collections.abc import Sized
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tsfb.base.data_module.dataloader.base_deep_loader import BaseDeepDataLoader
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)

ASTGCRNSplits = Tuple[
    Dict[str, np.ndarray],  # x_splits
    Dict[str, np.ndarray],  # y_splits
    List[str],  # nodes
    Optional[np.ndarray],  # od_tensor
]


class ASTGCRNDataLoader(BaseDeepDataLoader):
    """
    DataLoader specialized for ASTGCRN/ASTGCN models.

    - Input: output of `ASTGCRNPreprocess.fit(...)`
        (x_splits_norm, y_splits_raw, nodes, od_tensor)
    - Output: train/val/test DataLoaders with:
        X: (B, N, F, t_in)
        Y: (B, N, T_out)

    Additionally retains metadata for adapters:
        - mean/std (z-score) for reuse by ASTGCNAdapter
        - n_nodes, feat_dim, t_in
        - od_tensor (adjacency/OD)
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

        # metadata for ASTGCNAdapter
        self.nodes: List[str] = []
        self.od_tensor: Optional[np.ndarray] = None

        self.mean: Optional[np.ndarray] = None  # shape (1,1,1,1)
        self.std: Optional[np.ndarray] = None  # shape (1,1,1,1)

        self.n_nodes: Optional[int] = None
        self.feat_dim: Optional[int] = None
        self.t_in: Optional[int] = None

        # preserve val_target_tensor in legacy format (optional)
        self.val_target_tensor: Optional[Tensor] = None

    def build_from_splits(  # pylint: disable=too-many-locals
        self,
        splits: ASTGCRNSplits,
        *,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> Tuple[
        DataLoader[Tuple[Tensor, ...]],
        DataLoader[Tuple[Tensor, ...]],
        DataLoader[Tuple[Tensor, ...]],
    ]:
        """
        Build train/val/test DataLoaders from the output of ASTGCRNPreprocess.fit(...).

        Args:
            splits: (x_splits, y_splits, nodes, od_tensor)
            mean, std: z-score parameters (broadcastable shape 1x1x1x1).

        Returns:
            (train_loader, val_loader, test_loader)
        """
        x_splits, y_splits, nodes, od_tensor = splits

        self.nodes = nodes
        self.od_tensor = od_tensor
        self.mean = mean
        self.std = std

        (
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
        ) = self._extract_xy_splits(x_splits, y_splits)

        # infer meta shape from training data when available
        if train_x.size > 0:
            _, n_nodes, feat_dim, t_in = train_x.shape
            self.n_nodes = int(n_nodes)
            self.feat_dim = int(feat_dim)
            self.t_in = int(t_in)
        else:
            logger.warning("ASTGCRNDataLoader: train_x is empty, cannot infer shape.")

        # build DataLoaders
        self.train_loader = self._to_loader_supervised(train_x, train_y, shuffle=True)
        self.val_loader = self._to_loader_supervised(val_x, val_y, shuffle=False)
        self.test_loader = self._to_loader_supervised(test_x, test_y, shuffle=False)

        # preserve validation target tensor in legacy format (optional)
        if val_y.size > 0:
            self.val_target_tensor = torch.as_tensor(
                val_y, dtype=torch.float32, device=self.device
            )
        else:
            self.val_target_tensor = None

        if self.train_loader is not None:
            sized_dataset = cast(Sized, self.train_loader.dataset)
            if len(sized_dataset) > 0:
                x0, y0 = self.train_loader.dataset[0]
                logger.info(
                    "ASTGCRNDataLoader: sample X shape = %s, Y shape = %s",
                    tuple(x0.shape),
                    tuple(y0.shape),
                )
            else:
                logger.warning("ASTGCRNDataLoader: empty train dataset.")
        else:
            logger.warning("ASTGCRNDataLoader: train_loader is None.")

        return self.train_loader, self.val_loader, self.test_loader

    def get_adj_matrix(self) -> Optional[np.ndarray]:
        """Return adjacency / OD tensor."""
        return self.od_tensor
