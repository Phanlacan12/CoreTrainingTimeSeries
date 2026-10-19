"""
Random seed utilities for reproducible experiments.

This module provides functions to set random seeds for Python, NumPy, and PyTorch
in order to ensure reproducibility of experiments and results.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def fix_random_seed(seed: Optional[int] = 2021) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU only).

    Args:
        seed (Optional[int]): The random seed to use. If None, no action is taken.
    """
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def fix_all_random_seed(seed: Optional[int] = 2021) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU and CUDA),
    and configure PyTorch for deterministic behavior.

    Args:
        seed (Optional[int]): The random seed to use. If None, no action is taken.
    """
    if seed is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ["PYTHONHASHSEED"] = str(1)
