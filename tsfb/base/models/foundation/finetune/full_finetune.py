from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tsfb.base.models.foundation.finetune.base_finetune import FineTuner

if TYPE_CHECKING:
    from tsfb.base.schema.foundation_config import FineTuneConfig


class FullFineTuner(FineTuner):
    """
    Full-parameter fine-tuning strategy with gradient checkpointing,
    LR scheduling, and optional early stopping.

    All model parameters are trained. More expressive than LoRA but
    requires more GPU memory and data.
    """

    def prepare_model(self, model: nn.Module) -> nn.Module:
        for param in model.parameters():
            param.requires_grad = True
        if hasattr(model, "gradient_checkpointing_enable"):
            getattr(model, "gradient_checkpointing_enable")()
        return model

    def train(  # pylint: disable=too-many-locals
        self,
        model: nn.Module,
        train_data: np.ndarray,
        config: "FineTuneConfig",
        val_data: Optional[np.ndarray] = None,
    ) -> nn.Module:
        model = self.prepare_model(model)
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = config.epochs * max(1, len(train_data) // config.batch_size)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        criterion, early_stopping = self._prepare_training_state(val_data, config)

        model.train()

        for _ in range(config.epochs):
            for i in range(0, len(train_data), config.batch_size):
                batch = torch.tensor(
                    train_data[i : i + config.batch_size], dtype=torch.float32
                )
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            if self._should_stop_after_epoch(model, early_stopping):
                break

        if config.checkpoint_dir and val_data is not None:
            try:
                model = self.load_checkpoint(model, config.checkpoint_dir)
            except FileNotFoundError:
                pass

        model.eval()
        return model
