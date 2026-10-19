from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device, non_blocking=True)


def _standardize_input_window(
    x: np.ndarray | torch.Tensor, n_nodes: int, feat_dim: int, t_in: int
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = x
    if arr.ndim == 3 and arr.shape == (n_nodes, feat_dim, t_in):
        pass
    elif arr.ndim == 3 and arr.shape == (t_in, n_nodes, feat_dim):
        arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(
            f"series window must be (N,F,t_in) or (t_in,N,F), got {arr.shape}"
        )
    arr = arr.astype(np.float32, copy=False)[None, ...]
    return torch.from_numpy(arr)


@dataclass
class _BestState:
    epoch: int = -1
    val_loss: float = float("inf")
    state_dict: Optional[Dict[str, Any]] = None
