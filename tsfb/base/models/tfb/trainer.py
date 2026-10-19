from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from tsfb.base.schema.tfb_family import ModelFamily


@dataclass
class TrainingConfig:
    """Shared optimizer and device settings for lightweight TFB training."""

    n_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: Optional[int] = None
    device: str = "cpu"


def _resolve_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Model output must be a Tensor, got {type(output).__name__}")
    return output


def _align_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3 and target.ndim == 3 and pred.shape[1] != target.shape[1]:
        pred = pred[:, -target.shape[1] :, :]
    if pred.ndim == 3 and target.ndim == 3 and pred.shape[-1] != target.shape[-1]:
        pred = pred[..., : target.shape[-1]]
    return pred


def _forward_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    family: ModelFamily,
) -> torch.Tensor:
    if family == ModelFamily.SEQ2SEQ:
        return _resolve_output(
            model(
                batch["x_enc"],
                batch["x_mark_enc"],
                batch["x_dec"],
                batch["x_mark_dec"],
            )
        )
    if family == ModelFamily.SINGLE_INPUT:
        return _resolve_output(model(batch["x"]))
    if family == ModelFamily.MASKED_DECAY:
        return _resolve_output(
            model(batch["x"], batch["m"], batch["d"], batch["x_last"])
        )
    raise ValueError(f"Unsupported family {family!r}")


def _loss_fn(name: str) -> nn.Module:
    if name.lower() == "mae":
        return nn.L1Loss()
    return nn.MSELoss()


# pylint: disable-next=too-many-arguments
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    family: ModelFamily,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        target = batch["y"]
        pred = _forward_batch(model, batch, family)
        pred = _align_prediction(pred, target)
        loss = criterion(pred, target)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = int(target.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    if total_items == 0:
        return 0.0
    return total_loss / total_items


# pylint: disable-next=too-many-arguments
def fit_model(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    family: ModelFamily,
    config: TrainingConfig,
    loss_name: str = "mse",
) -> nn.Module:
    """Train a model with a small shared PyTorch loop."""

    device = torch.device(config.device)
    model.to(device)
    criterion = _loss_fn(loss_name).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    for _ in range(config.n_epochs):
        _run_epoch(model, train_loader, family, criterion, optimizer, device)
        if val_loader is None:
            continue

        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, family, criterion, None, device)
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if config.patience is not None and stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_batches(
    model: nn.Module,
    *,
    loader: DataLoader,
    family: ModelFamily,
    device: str,
) -> torch.Tensor:
    """Run batched inference and return concatenated predictions."""

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(torch_device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            pred = _align_prediction(_forward_batch(model, batch, family), batch["y"])
            outputs.append(pred.detach().cpu())
    if not outputs:
        return torch.empty((0,))
    return torch.cat(outputs, dim=0)
