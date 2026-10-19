from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from tsfb.base.models.foundation.finetune.base_finetune import FineTuner

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

if TYPE_CHECKING:
    from tsfb.base.schema.foundation_config import FineTuneConfig


class LoRAFineTuner(FineTuner):
    """
    Fine-tuning strategy using LoRA (Low-Rank Adaptation) via the PEFT library.

    Only the low-rank adapter weights are trained; the backbone is frozen.
    Suitable for efficient domain adaptation with small datasets.

    Required extra dependency:
        pip install peft
    """

    def _require_peft(self) -> None:
        if not _PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA fine-tuning.\n"
                "Install with: pip install peft"
            )

    def prepare_model(self, model: nn.Module) -> nn.Module:
        self._require_peft()
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
        )
        return get_peft_model(model, lora_config)

    def prepare_model_with_config(
        self, model: nn.Module, config: "FineTuneConfig"
    ) -> nn.Module:
        """Apply LoRA adapters using rank/alpha/dropout from *config*."""
        self._require_peft()
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q", "v"],
        )
        return get_peft_model(model, lora_config)

    def train(  # pylint: disable=too-many-locals
        self,
        model: nn.Module,
        train_data: np.ndarray,
        config: "FineTuneConfig",
        val_data: Optional[np.ndarray] = None,
    ) -> nn.Module:
        peft_model = self.prepare_model_with_config(model, config)
        optimizer = AdamW(
            peft_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = config.epochs * max(1, len(train_data) // config.batch_size)
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps,
        )
        criterion, early_stopping = self._prepare_training_state(val_data, config)

        peft_model.train()

        for _ in range(config.epochs):
            batch_losses = []
            for i in range(0, len(train_data), config.batch_size):
                batch = torch.tensor(
                    train_data[i : i + config.batch_size], dtype=torch.float32
                )
                optimizer.zero_grad()
                output = peft_model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_losses.append(loss.item())

            if self._should_stop_after_epoch(peft_model, early_stopping):
                break

        peft_model.eval()
        return peft_model
