from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsfb.base.data_module.preprocess_model.components.torch_collate import (
    collate_tensor_dict,
)


class TensorDictDataset(Dataset):
    """A minimal dataset wrapper around window sample dictionaries."""

    def __init__(self, samples: Dict[str, Any]):
        self.samples = samples
        data_keys = [
            key
            for key, value in samples.items()
            if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0
        ]
        self._data_keys = data_keys
        self._length = (
            int(len(samples[data_keys[0]]))
            if data_keys
            else int(samples.get("num_samples", 0))
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}
        for key, value in self.samples.items():
            if isinstance(value, np.ndarray):
                item[key] = torch.as_tensor(value[idx], dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                item[key] = value[idx]
            elif isinstance(value, list) and len(value) == self._length:
                item[key] = value[idx]
        return item


def build_loader(
    samples: Dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Build a PyTorch DataLoader from a tensor-dictionary sample payload."""
    dataset = TensorDictDataset(samples)
    effective_drop_last = drop_last and len(dataset) >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and len(dataset) > 0,
        drop_last=effective_drop_last,
        num_workers=num_workers,
        collate_fn=collate_tensor_dict,
    )
