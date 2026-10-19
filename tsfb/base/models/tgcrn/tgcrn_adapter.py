from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tsfb.base.data_module.dataloader.astgcrn_loader import ASTGCRNDataLoader
from tsfb.base.data_module.preprocess_model.astgcrn_preprocess import ASTGCRNPreprocess
from tsfb.base.models.base_model import BatchMaker, ModelBase
from tsfb.base.origin_code.tgcrn.model.TGCRN import TGCRN as OriginTGCRN


@dataclass
class _BestState:
    loss: float = float("inf")
    state_dict: Optional[Dict[str, torch.Tensor]] = None


class TGCRNAdapter(ModelBase):
    """Dedicated adapter for TGCRN using the ASTGCRN graph pipeline."""

    def __init__(self, **config: Any):
        self.config = dict(config)
        device_name = self.config.get(
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
        self.device = torch.device(str(device_name))
        self._model: Optional[nn.Module] = None
        self._preprocess: Optional[ASTGCRNPreprocess] = None
        self._target_cols: list[str] = []
        self._n_nodes: Optional[int] = None
        self._feat_dim: Optional[int] = None
        self._seq_len = int(self.config["input_chunk_length"])
        self._pred_len = int(self.config["output_chunk_length"])

    @property
    def model_name(self) -> str:
        return "tgcrn.TGCRN"

    def _build_preprocess_config(self, train_ratio_in_tv: float) -> Dict[str, Any]:
        val_ratio = max(0.0, 1.0 - float(train_ratio_in_tv))
        return {
            "num_of_hours": int(self.config["input_chunk_length"]),
            "num_of_days": int(self.config.get("num_of_days", 0)),
            "num_of_weeks": int(self.config.get("num_of_weeks", 0)),
            "num_for_predict": int(self.config["output_chunk_length"]),
            "points_per_hour": int(self.config.get("points_per_hour", 1)),
            "normalize": self.config.get("normalize", True),
            "split_ratio": {
                "train": float(train_ratio_in_tv),
                "val": val_ratio,
                "test": 0.0,
            },
        }

    def _build_args(self, *, num_targets: int, num_nodes: int) -> SimpleNamespace:
        input_dim = int(self.config.get("input_dim", max(1, num_targets // num_nodes)))
        output_dim = int(self.config.get("output_dim", input_dim))
        return SimpleNamespace(
            num_nodes=num_nodes,
            input_dim=input_dim,
            time_dim=int(self.config.get("time_dim", 8)),
            rnn_units=int(
                self.config.get(
                    "rnn_units",
                    self.config.get("hidden_size", self.config.get("d_model", 64)),
                )
            ),
            output_dim=output_dim,
            horizon=self._pred_len,
            num_layers=int(self.config.get("num_layers", 1)),
            embed_dim=int(self.config.get("embed_dim", self.config.get("d_model", 64))),
            cheb_k=int(self.config.get("cheb_k", 2)),
            graph_direction=self.config.get("graph_direction", "symm"),
            time_embedding=bool(self.config.get("time_embedding", False)),
            od_flag=bool(self.config.get("od_flag", False)),
            time_station=bool(self.config.get("time_station", False)),
            period=bool(self.config.get("period", False)),
            Seq_Dec=bool(self.config.get("Seq_Dec", False)),
            constrative_time=bool(self.config.get("constrative_time", False)),
            node_mode=self.config.get("node_mode", "direct"),
            num_time=int(self.config.get("num_time", 0)),
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type for TGCRNAdapter: {type(batch)!r}")

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Call forecast_fit() before forecast().")
        pred = self._model(x.permute(0, 3, 1, 2))
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim == 4:
            pred = pred[..., 0]
        if pred.ndim != 3:
            raise ValueError(f"Unexpected TGCRN output shape: {tuple(pred.shape)}")
        return pred.permute(0, 2, 1)

    @staticmethod
    def _align_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] != target.shape[1]:
            pred = pred[:, -target.shape[1] :, :]
        if pred.shape[-1] != target.shape[-1]:
            pred = pred[..., : target.shape[-1]]
        return pred

    def _loss_fn(self) -> nn.Module:
        if str(self.config.get("loss_name", "mse")).lower() == "mae":
            return nn.L1Loss()
        return nn.MSELoss()

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: nn.Module,
    ) -> float:
        if self._model is None:
            raise RuntimeError("Model is not initialized.")
        training = optimizer is not None
        self._model.train(training)
        total_loss = 0.0
        total_items = 0

        for batch in loader:
            x, y = self._unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self._align_prediction(self._forward(x), y)
            loss = criterion(pred, y)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = int(y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        return 0.0 if total_items == 0 else total_loss / total_items

    def _fit_from_dataframe(
        self,
        train_data: pd.DataFrame,
        *,
        train_ratio_in_tv: float,
        loader: Any,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        nodes = list(getattr(loader, "nodes", []) or [])
        if not nodes:
            nodes = [str(col) for col in train_data.columns]
        od_tensor = getattr(loader, "od_tensor", None)

        self._target_cols = list(train_data.columns)
        self._preprocess = ASTGCRNPreprocess(
            self._build_preprocess_config(train_ratio_in_tv)
        )
        splits = self._preprocess.fit((train_data.copy(), nodes, od_tensor))
        graph_loader = ASTGCRNDataLoader(
            batch_size=int(self.config.get("batch_size", 32)),
            device=self.device,
            drop_last=False,
            num_workers=int(self.config.get("num_workers", 0)),
        )
        train_loader, val_loader, _ = graph_loader.build_from_splits(
            splits,
            mean=self._preprocess.normalization_stats[0],
            std=self._preprocess.normalization_stats[1],
        )
        self._n_nodes = graph_loader.n_nodes or len(nodes)
        self._feat_dim = graph_loader.feat_dim or max(
            1, len(self._target_cols) // max(1, self._n_nodes)
        )
        return train_loader, val_loader

    # pylint: disable-next=too-many-locals
    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs: Any,
    ) -> "TGCRNAdapter":
        del covariates
        loader = kwargs.get("loader")
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("TGCRNAdapter expects train_data as a pandas.DataFrame.")
        if loader is None:
            raise ValueError("TGCRNAdapter requires graph metadata via loader=...")

        train_loader, val_loader = self._fit_from_dataframe(
            train_data,
            train_ratio_in_tv=train_ratio_in_tv,
            loader=loader,
        )
        if self._n_nodes is None:
            raise RuntimeError("Failed to infer TGCRN num_nodes.")

        args = self._build_args(
            num_targets=len(self._target_cols),
            num_nodes=self._n_nodes,
        )
        self._model = OriginTGCRN(args).to(self.device)

        criterion = self._loss_fn().to(self.device)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

        best = _BestState()
        n_epochs = int(self.config.get("n_epochs", 1))
        patience = self.config.get("patience")
        stale_epochs = 0

        for _ in range(n_epochs):
            self._run_epoch(train_loader, optimizer=optimizer, criterion=criterion)
            if val_loader is None:
                continue
            with torch.no_grad():
                val_loss = self._run_epoch(
                    val_loader,
                    optimizer=None,
                    criterion=criterion,
                )
            if val_loss < best.loss:
                best = _BestState(
                    loss=val_loss,
                    state_dict={
                        key: value.detach().cpu().clone()
                        for key, value in self._model.state_dict().items()
                    },
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience is not None and stale_epochs >= int(patience):
                    break

        if best.state_dict is not None:
            self._model.load_state_dict(best.state_dict)
        self._model.eval()
        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        del covariates
        if self._model is None or self._preprocess is None:
            raise RuntimeError("Call forecast_fit() before forecast().")
        if horizon != self._pred_len:
            raise ValueError(f"Expected horizon={self._pred_len}, got {horizon}.")
        if self._n_nodes is None or self._feat_dim is None:
            raise RuntimeError("TGCRN shape metadata is not initialized.")
        if len(series) < self._seq_len:
            raise ValueError(
                f"TGCRN requires at least {self._seq_len} rows "
                f"for inference, got {len(series)}."
            )

        values = series[self._target_cols].to_numpy(dtype=np.float32, copy=False)
        expected_width = self._n_nodes * self._feat_dim
        if values.shape[1] != expected_width:
            raise ValueError(
                "TGCRN inference width mismatch: expected "
                f"{expected_width} columns, got {values.shape[1]}."
            )

        window = values[-self._seq_len :]
        x = window.reshape(self._seq_len, self._n_nodes, self._feat_dim)
        x = np.transpose(x, (1, 2, 0))[None, ...]
        x = self._preprocess.normalize_inputs(x)

        x_tensor = torch.from_numpy(x).to(self.device)
        self._model.eval()
        with torch.no_grad():
            pred = self._forward(x_tensor).detach().cpu().numpy()[0]
        if pred.shape[0] != horizon:
            pred = pred[:horizon]
        return pred

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        **kwargs: Any,
    ) -> np.ndarray:
        """Expose the batch-forecast interface explicitly for TGCRN."""
        raise NotImplementedError(
            "TGCRNAdapter does not yet implement batch_forecast()."
        )
