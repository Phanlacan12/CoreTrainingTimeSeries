from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim

from tsfb.base.data_module.dataloader.astgcrn_loader import ASTGCRNDataLoader
from tsfb.base.models.astgcn.astgcn_utils import (
    _BestState,
    _standardize_input_window,
    _to_device,
)
from tsfb.base.models.base_model import BatchMaker, ModelBase
from tsfb.base.origin_code.astgcrn.lib.metrics import masked_mae, masked_mse
from tsfb.base.origin_code.astgcrn.lib.utils import get_adjacency_matrix
from tsfb.base.origin_code.astgcrn.model.ASTGCN_r import make_model


class ASTGCNAdapter(ModelBase):
    """
    ASTGCN Adapter (uses a data loader provided by the user).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            "cuda:0"
            if torch.cuda.is_available() and config.get("use_cuda", True)
            else "cpu"
        )

        # Architecture / training hyperparams
        self.num_for_predict = int(config["num_for_predict"])
        self.len_input = int(config["len_input"])
        self.nb_block = int(config["nb_block"])
        self.in_channels = int(config["in_channels"])
        self.cheb_k = int(config["K"])
        self.nb_chev_filter = int(config["nb_chev_filter"])
        self.nb_time_filter = int(config["nb_time_filter"])
        self.time_strides = int(config.get("time_strides", 1))
        self.learning_rate = float(config["learning_rate"])
        self.epochs = int(config["epochs"])
        self.start_epoch = int(config.get("start_epoch", 0))
        self.batch_size = int(config["batch_size"])
        self.loss_function = config.get("loss_function", "mae")
        self.missing_value = float(config.get("missing_value", 0.0))

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._n_nodes: Optional[int] = None
        self._feat_dim: Optional[int] = None
        self._t_in: Optional[int] = None

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._val_target = None

        self._adj_mx: Optional[np.ndarray] = None
        self._net: Optional[nn.Module] = None
        self._best = _BestState()

        self.log_dir = config.get("log_dir")
        self.print_every = int(config.get("print_every", 100))
        self.val_print_every = int(config.get("val_print_every", 100))
        self.save_best = bool(config.get("save_best", False))  # if you want to save .pt
        self.params_path = config.get("params_path")
        self._writer = None

    @property
    def model_name(self) -> str:
        return "ASTGCNAdapter"

    def _coerce_to_adj_square(self, adj, n_nodes: int) -> np.ndarray | None:
        if adj is None:
            return None
        arr = np.asarray(adj)

        # Case 1: (T, N, N) -> collapse over time
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
            arr = np.asarray(arr.any(axis=0), dtype=np.float32)

        # Case 2: (N, N, K) -> max over K
        if arr.ndim == 3 and arr.shape[0] == arr.shape[1]:
            arr = np.asarray(arr.max(axis=-1) > 0, dtype=np.float32)

        # Case 3: (E,3) or (E,2) edge list -> build A(N,N)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1] and arr.shape[1] in (2, 3):
            src = arr[:, 0].astype(int)
            dst = arr[:, 1].astype(int)
            # Check whether indices are valid
            if (
                src.max() >= n_nodes
                or dst.max() >= n_nodes
                or src.min() < 0
                or dst.min() < 0
            ):
                raise ValueError(
                    f"Edge list node id out of range "
                    f"(max src={src.max()}, max dst={dst.max()}, n_nodes={n_nodes}). "
                    "If ids in the CSV are not 0..N-1, provide 'id_filename' to map, "
                    "or set training_cfg['adj_filename'] to use get_adjacency_matrix()."
                )
            adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            adj_matrix[src, dst] = 1.0
            arr = adj_matrix  # continue to symmetry/self-loop fixing step

        # Case 4: (N,N)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            pass
        else:
            raise ValueError(f"Cannot coerce adj shape {arr.shape} to square (N,N)")

        # Final normalization: binary, symmetrize (if undirected), remove self-loop
        arr = (arr > 0).astype(np.float32)
        if self.config.get("make_undirected", True):
            arr = np.maximum(arr, arr.T)
        if self.config.get("remove_self_loop", True):
            np.fill_diagonal(arr, 0.0)
        return arr

    def _unpack_train_data(self, train_data):
        """Unpack `train_data` which can be ASTGCRNDataLoader, tuple, or dict.
        Populates internal attributes: loaders, mean/std,
        n_nodes/feat_dim/t_in and adj (if provided).
        """
        # Case 1: ASTGCRNDataLoader
        if isinstance(train_data, ASTGCRNDataLoader):
            self._train_loader = train_data.train_loader
            self._val_loader = train_data.val_loader
            self._test_loader = train_data.test_loader
            self._val_target = train_data.val_target_tensor

            # mean/std
            if train_data.mean is not None:
                self._mean = train_data.mean
            else:
                self._mean = np.array(0.0, dtype=np.float32).reshape((1, 1, 1, 1))

            if train_data.std is not None:
                self._std = train_data.std
            else:
                self._std = np.array(1.0, dtype=np.float32).reshape((1, 1, 1, 1))

            self._n_nodes = (
                int(train_data.n_nodes) if train_data.n_nodes is not None else None
            )
            self._feat_dim = (
                int(train_data.feat_dim) if train_data.feat_dim is not None else None
            )
            self._t_in = int(train_data.t_in) if train_data.t_in is not None else None

            if self._n_nodes is None or self._feat_dim is None or self._t_in is None:
                loader = self._train_loader or self._val_loader or self._test_loader
                if loader is None:
                    raise RuntimeError(
                        "ASTGCRNDataLoader has no non-empty loader to infer shapes."
                    )
                sample_x, _ = next(iter(loader))
                _, n_nodes, feat_dim, t_in = sample_x.shape
                self._n_nodes = int(n_nodes)
                self._feat_dim = int(feat_dim)
                self._t_in = int(t_in)

            # adjacency
            self._adj_mx = train_data.get_adj_matrix()
            if self._adj_mx is not None:
                self._adj_mx = self._coerce_to_adj_square(self._adj_mx, self._n_nodes)

        # Case 2: tuple
        elif isinstance(train_data, tuple):
            (
                self._train_loader,
                self._val_loader,
                self._test_loader,
                self._mean,
                self._std,
                self._n_nodes,
                self._feat_dim,
                self._t_in,
                *maybe_adj,
            ) = train_data
            self._adj_mx = maybe_adj[0] if maybe_adj else None

        # Case 3: dict
        elif isinstance(train_data, dict):
            self._train_loader = train_data["train_loader"]
            self._val_loader = train_data.get("val_loader")
            self._test_loader = train_data.get("test_loader")
            self._mean = train_data["mean"]
            self._std = train_data["std"]
            self._n_nodes = int(train_data["n_nodes"])
            self._feat_dim = int(train_data["feat_dim"])
            self._t_in = int(train_data["t_in"])
            self._adj_mx = train_data.get("adj_mx")
            self._val_target = train_data.get("val_target_tensor")
        else:
            raise TypeError("Unsupported train_data type for ASTGCNAdapter.")

    def _load_adj_if_missing(self):
        """Load adjacency matrix from configured file if it's not already set."""
        if self._adj_mx is not None:
            return
        adj_filename = self.config.get("adj_filename")
        num_of_vertices = int(self.config.get("num_of_vertices", self._n_nodes))
        id_filename = self.config.get("id_filename")
        if adj_filename is None:
            raise ValueError("adj_mx is None and config.adj_filename is not set.")
        self._adj_mx, _ = get_adjacency_matrix(
            adj_filename, num_of_vertices, id_filename
        )

    def _build_model_and_criterion(self):
        """Create model, optimizer, loss criterion and masked_flag.
        Returns (optimizer, criterion, masked_flag)."""
        net = make_model(
            self.device,
            self.nb_block,
            self.in_channels,
            self.cheb_k,
            self.nb_chev_filter,
            self.nb_time_filter,
            self.time_strides,
            self._adj_mx,
            self.num_for_predict,
            self.len_input,
            self._n_nodes,
        ).eval()

        self._net = net

        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)

        masked_flag = self.loss_function in {"masked_mse", "masked_mae"}
        if self.loss_function == "masked_mse":
            criterion = masked_mse
        elif self.loss_function == "masked_mae":
            criterion = masked_mae
        elif self.loss_function == "rmse":
            criterion = nn.MSELoss().to(self.device)
            masked_flag = False
        else:
            criterion = nn.L1Loss().to(self.device)
            masked_flag = False

        return optimizer, criterion, masked_flag

    def _evaluate_val(self, criterion, masked_flag, epoch) -> float:
        """Evaluate validation loader and return mean loss."""
        if self._val_loader is None:
            return float("inf")
        net = self._net
        assert net is not None
        net.eval()
        losses = []
        with torch.no_grad():
            for bi, (x, y) in enumerate(self._val_loader):
                x = _to_device(x, self.device)
                y = _to_device(y, self.device)
                out = net(x)
                loss = (
                    criterion(out, y, self.missing_value)
                    if masked_flag
                    else criterion(out, y)
                )
                losses.append(float(loss.item()))
                if bi % self.val_print_every == 0:
                    print(
                        f"[VAL] batch {bi + 1}/{len(self._val_loader)}, "
                        f"loss: {loss.item():.4f}"
                    )
        val = float(np.mean(losses)) if losses else float("inf")
        if self._writer:
            self._writer.add_scalar("validation_loss", val, epoch)
        return val

    def _train_loop(
        self, optimizer, criterion, masked_flag, global_step: int = 0
    ) -> int:
        """Run epochs: validation, training, checkpointing.
        Returns final global_step."""
        # ensure _net is initialized and use local variable for type-checkers
        net = self._net
        assert net is not None
        net.train()
        for epoch in range(self.start_epoch, self.epochs):
            # validate first (like the original script)
            val_loss = self._evaluate_val(criterion, masked_flag, epoch)
            if val_loss < self._best.val_loss:
                # net is guaranteed not None here
                self._best = _BestState(
                    epoch=epoch,
                    val_loss=val_loss,
                    state_dict={
                        k: v.detach().cpu().clone() for k, v in net.state_dict().items()
                    },
                )
                print(f"[CKPT] epoch {epoch}: new best val_loss={val_loss:.6f}")
                if self.save_best and self.params_path:
                    os.makedirs(self.params_path, exist_ok=True)
                    torch.save(
                        net.state_dict(),
                        os.path.join(self.params_path, f"epoch_{epoch}.pt"),
                    )
                    best_epoch_path = os.path.join(
                        self.params_path, f"epoch_{epoch}.pt"
                    )
                    print(f"[CKPT] saved to {best_epoch_path}")

            if self._train_loader is None:
                continue

            for bi, (x, y) in enumerate(self._train_loader):
                x = _to_device(x, self.device)
                y = _to_device(y, self.device)
                optimizer.zero_grad(set_to_none=True)
                out = net(x)
                loss = (
                    criterion(out, y, self.missing_value)
                    if masked_flag
                    else criterion(out, y)
                )
                loss.backward()
                optimizer.step()
                global_step += 1

                # print & tensorboard
                if bi % self.print_every == 0:
                    print(
                        f"[TRAIN] epoch {epoch} step {global_step} "
                        f"batch {bi}/{len(self._train_loader)} "
                        f"loss: {loss.item():.4f}"
                    )
                if self._writer:
                    self._writer.add_scalar(
                        "training_loss", float(loss.item()), global_step
                    )
        return global_step

    def forecast_fit(
        self,
        train_data,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ASTGCNAdapter":
        """
        train_data:
          - Case 1: SpatioDataLoader (already built)
          - Case 2: tuple (train_loader, val_loader, test_loader,
            mean, std, n_nodes, feat_dim, t_in, adj_mx?)
          - Case 3: dict with keys similar to Case 2
        """
        self._unpack_train_data(train_data)
        self._load_adj_if_missing()
        if self.log_dir:
            self._writer = SummaryWriter(logdir=self.log_dir, flush_secs=5)
        global_step = 0
        optimizer, criterion, masked_flag = self._build_model_and_criterion()
        global_step = self._train_loop(optimizer, criterion, masked_flag, global_step)
        if self._best.state_dict is not None:
            net = self._net
            assert net is not None
            net.load_state_dict(self._best.state_dict, strict=True)
            net.eval()
        return self

    @torch.no_grad()
    def forecast(
        self, horizon: int, series, *, covariates: Optional[dict] = None
    ) -> np.ndarray:
        """forecast a single input window."""
        if self._net is None or self._mean is None or self._std is None:
            raise RuntimeError("Call forecast_fit() before forecast().")
        if horizon != self.num_for_predict:
            raise ValueError(
                f"horizon {horizon} != num_for_predict {self.num_for_predict}"
            )
        n_nodes = self._n_nodes
        feat_dim = self._feat_dim
        t_in = self._t_in
        assert n_nodes is not None and feat_dim is not None and t_in is not None

        x = _standardize_input_window(series, n_nodes, feat_dim, t_in)
        x = _to_device(x.float(), self.device)
        mean = torch.from_numpy(self._mean.astype(np.float32)).to(self.device)
        std = torch.from_numpy(self._std.astype(np.float32)).to(self.device)
        net = self._net
        assert net is not None
        y_norm = net((x - mean) / std)  # (1,N,T_out)
        y = y_norm * std + mean  # inverse
        return y.detach().cpu().numpy()[0].astype(np.float32)

    @torch.no_grad()
    def batch_forecast(  # pylint: disable=too-many-locals
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """batch forecast using a BatchMaker to create batches."""
        if self._net is None or self._mean is None or self._std is None:
            raise RuntimeError("Call forecast_fit() before batch_forecast().")
        if horizon != self.num_for_predict:
            raise ValueError(
                f"horizon {horizon} != num_for_predict {self.num_for_predict}"
            )

        # ensure shape metadata present and narrow types for the type-checker
        n_nodes = self._n_nodes
        feat_dim = self._feat_dim
        t_in = self._t_in
        assert n_nodes is not None and feat_dim is not None and t_in is not None

        batch_size = int(kwargs.get("batch_size", self.batch_size))
        win_size = int(kwargs.get("win_size", t_in))
        pack = batch_maker.make_batch(batch_size, win_size)
        series = pack["series"]  # (B,N,F,t_in) or (B,t_in,N,F)

        arr = (
            series.detach().cpu().numpy().astype(np.float32, copy=False)
            if isinstance(series, torch.Tensor)
            else np.asarray(series, dtype=np.float32)
        )

        if arr.ndim != 4:
            raise ValueError(f"series must be 4D, got {arr.shape}")
        if arr.shape[1:] == (n_nodes, feat_dim, t_in):
            x = arr
        elif arr.shape[1:] == (t_in, n_nodes, feat_dim):
            x = np.transpose(arr, (0, 2, 3, 1))
        else:
            raise ValueError(f"series wrong shape {arr.shape}")

        x_t = torch.from_numpy(x).to(self.device)
        mean = torch.from_numpy(self._mean.astype(np.float32)).to(self.device)
        std = torch.from_numpy(self._std.astype(np.float32)).to(self.device)
        net = self._net
        assert net is not None
        y = net((x_t - mean) / std) * std + mean
        return y.detach().cpu().numpy().astype(np.float32)
