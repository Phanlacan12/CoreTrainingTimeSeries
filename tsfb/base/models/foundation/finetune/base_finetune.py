from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from tsfb.base.schema.foundation_config import FineTuneConfig


@dataclass
class _EarlyStoppingState:
    val_data: np.ndarray
    config: "FineTuneConfig"
    criterion: nn.Module
    best_val_loss: float = float("inf")
    patience_counter: int = 0


class FineTuner(abc.ABC):
    """
    Abstract base class for fine-tuning strategy implementations.

    Concrete subclasses (LoRAFineTuner, FullFineTuner) implement the
    prepare_model / train methods. Checkpoint I/O is provided here.
    """

    @abc.abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Apply model modifications required before training."""

    @abc.abstractmethod
    def train(
        self,
        model: nn.Module,
        train_data: np.ndarray,
        config: "FineTuneConfig",
        val_data: Optional[np.ndarray] = None,
    ) -> nn.Module:
        """Run the fine-tuning loop and return the updated model."""

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        """Persist model state_dict to *path*/model.pt."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        """Load state_dict from *path*/model.pt into *model* and return it."""
        ckpt = Path(path) / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        return model

    def _make_early_stopping_state(
        self,
        val_data: Optional[np.ndarray],
        config: "FineTuneConfig",
        criterion: nn.Module,
    ) -> Optional[_EarlyStoppingState]:
        """Create early-stopping state when validation checks are enabled."""
        if val_data is None or not config.early_stopping_patience:
            return None
        return _EarlyStoppingState(val_data, config, criterion)

    def _prepare_training_state(
        self,
        val_data: Optional[np.ndarray],
        config: "FineTuneConfig",
    ) -> tuple[nn.Module, Optional[_EarlyStoppingState]]:
        """Build common loss and early-stopping state for a training run."""
        criterion = torch.nn.MSELoss()
        early_stopping = self._make_early_stopping_state(
            val_data,
            config,
            criterion,
        )
        return criterion, early_stopping

    def _run_epoch_val_check(
        self,
        model: nn.Module,
        state: _EarlyStoppingState,
    ) -> bool:
        """Evaluate val loss and apply early-stopping logic.
        Returns whether training should stop."""
        val_tensor = torch.tensor(state.val_data, dtype=torch.float32)
        with torch.no_grad():
            val_output = model(val_tensor)
            val_loss = state.criterion(val_output, val_tensor).item()
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.patience_counter = 0
            if state.config.checkpoint_dir:
                self.save_checkpoint(model, state.config.checkpoint_dir)
        else:
            state.patience_counter += 1
            patience = state.config.early_stopping_patience or 0
            if state.patience_counter >= patience:
                return True
        return False

    def _should_stop_after_epoch(
        self,
        model: nn.Module,
        state: Optional[_EarlyStoppingState],
    ) -> bool:
        """Run an epoch validation check if early stopping is enabled."""
        if state is None:
            return False
        return self._run_epoch_val_check(model, state)
