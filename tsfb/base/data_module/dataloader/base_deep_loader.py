from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


DeepSplits = Tuple[
    Dict[str, np.ndarray],  # x_splits: {"train": X_tr, "val": X_val, "test": X_te}
    Dict[str, np.ndarray],  # y_splits: {"train": Y_tr, "val": Y_val, "test": Y_te}
]

DeepInference = Tuple[
    np.ndarray,  # X_full
    Optional[np.ndarray],  # Optional Y_full (may be None)
]


class BaseDeepDataLoader(ABC):
    """
    Abstract base class for Torch DataLoaders used in deep learning.

    Goals:
    - Standardize the API for build_from_splits(...) for train/val/test
    - Be domain-agnostic (spatio, tabular, time-series, ...)
    """

    def __init__(
        self,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        *,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.drop_last = drop_last
        self.num_workers = num_workers

        self.train_loader: Optional[DataLoader[Tuple[Tensor, ...]]] = None
        self.val_loader: Optional[DataLoader[Tuple[Tensor, ...]]] = None
        self.test_loader: Optional[DataLoader[Tuple[Tensor, ...]]] = None
        self.inference_loader: Optional[DataLoader[Tuple[Tensor, ...]]] = None

    @abstractmethod
    def build_from_splits(
        self,
        splits: Any,
    ) -> Tuple[
        DataLoader[Tuple[Tensor, ...]],
        DataLoader[Tuple[Tensor, ...]],
        DataLoader[Tuple[Tensor, ...]],
    ]:
        """
        Create train/val/test DataLoaders from splits.

        In subclasses:
        - splits may be `DeepSplits` (x_splits, y_splits)
        - or `SpatioSplits` (x_splits, y_splits, nodes, od_tensor)
        """
        raise NotImplementedError

    def _to_loader_supervised(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        shuffle: bool,
    ) -> DataLoader[Tuple[Tensor, ...]]:
        """Helper: create a supervised DataLoader from numpy arrays x and y."""
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32).to(self.device),
            torch.tensor(y, dtype=torch.float32).to(self.device),
        )
        effective_shuffle = shuffle and len(ds) > 0
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=effective_shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def _to_loader_unsupervised(
        self,
        x: np.ndarray,
        *,
        shuffle: bool,
    ) -> DataLoader[Tuple[Tensor, ...]]:
        """Helper: create an unsupervised DataLoader (X only)."""
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32).to(self.device),
        )
        effective_shuffle = shuffle and len(ds) > 0
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=effective_shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def get_train_val_test_loaders(
        self,
    ) -> Tuple[
        Optional[DataLoader[Tuple[Tensor, ...]]],
        Optional[DataLoader[Tuple[Tensor, ...]]],
        Optional[DataLoader[Tuple[Tensor, ...]]],
    ]:
        """Return (train_loader, val_loader, test_loader)."""
        return self.train_loader, self.val_loader, self.test_loader

    def get_inference_loader(self) -> Optional[DataLoader[Tuple[Tensor, ...]]]:
        """Return the inference_loader (if built)."""
        return self.inference_loader

    def _extract_xy_splits(
        self,
        x_splits: Dict[str, np.ndarray],
        y_splits: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """Validate and extract (train/val/test) X, y from split dicts."""
        for k in ("train", "val", "test"):
            if k not in x_splits or k not in y_splits:
                raise KeyError(f"Missing split '{k}' in x_splits/y_splits.")

        train_x = x_splits["train"]
        train_y = y_splits["train"]
        val_x = x_splits["val"]
        val_y = y_splits["val"]
        test_x = x_splits["test"]
        test_y = y_splits["test"]

        return train_x, train_y, val_x, val_y, test_x, test_y
