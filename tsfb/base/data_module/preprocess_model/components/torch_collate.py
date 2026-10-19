from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_tensor_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack dictionary batches while keeping non-tensor values as lists."""

    if not batch:
        return {}

    out: Dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out
