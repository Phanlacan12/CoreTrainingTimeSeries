import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tsfb.base.data_module.dataloader.spatio_loader import SpatioTorchDataLoader


def make_spatio_splits(
    n_train: int = 10,
    n_val: int = 6,
    n_test: int = 4,
    n_nodes: int = 3,
    n_feat: int = 2,
    t_in: int = 5,
    t_out: int = 3,
):
    """Create simple dummy SpatioSplits with shapes:
    X: (B, N, F, T_in), Y: (B, N, F, T_out).
    """

    def make_xy(b: int):
        x = np.random.randn(b, n_nodes, n_feat, t_in).astype("float32")
        y = np.random.randn(b, n_nodes, n_feat, t_out).astype("float32")
        return x, y

    x_tr, y_tr = make_xy(n_train)
    x_val, y_val = make_xy(n_val)
    x_te, y_te = make_xy(n_test)

    x_splits = {"train": x_tr, "val": x_val, "test": x_te}
    y_splits = {"train": y_tr, "val": y_val, "test": y_te}

    nodes = [f"n{i}" for i in range(n_nodes)]
    od_tensor = np.random.randn(n_nodes, n_nodes).astype("float32")

    return x_splits, y_splits, nodes, od_tensor


def test_build_from_splits_basic_shapes_and_metadata():
    """build_from_splits must return DataLoaders with correct shapes and metadata."""
    splits = make_spatio_splits(
        n_train=12,
        n_val=8,
        n_test=6,
        n_nodes=2,
        n_feat=3,
        t_in=4,
        t_out=2,
    )
    loader_obj = SpatioTorchDataLoader(
        batch_size=4,
        device=torch.device("cpu"),
        drop_last=False,
        num_workers=0,
    )

    train_loader, val_loader, test_loader = loader_obj.build_from_splits(splits)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    x_splits, y_splits, nodes, od_tensor = splits

    assert len(train_loader.dataset) == x_splits["train"].shape[0]
    assert len(val_loader.dataset) == x_splits["val"].shape[0]
    assert len(test_loader.dataset) == x_splits["test"].shape[0]

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[1:] == torch.Size((2, 3, 4))
    assert batch_y.shape[1:] == torch.Size((2, 3, 2))
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.float32

    assert loader_obj.nodes == nodes
    assert loader_obj.get_adj_matrix() is loader_obj.od_tensor
    assert np.array_equal(loader_obj.od_tensor, od_tensor)


def test_build_from_splits_drop_last():
    """drop_last=True must drop the last incomplete batch."""
    splits = make_spatio_splits(
        n_train=10,
        n_val=5,
        n_test=5,
        n_nodes=2,
        n_feat=1,
        t_in=3,
        t_out=2,
    )
    loader_obj = SpatioTorchDataLoader(
        batch_size=4,
        device=torch.device("cpu"),
        drop_last=True,
        num_workers=0,
    )

    train_loader, _, _ = loader_obj.build_from_splits(splits)

    batches = list(train_loader)
    assert len(batches) == 2
    for bx, by in batches:
        assert bx.shape[0] == 4
        assert by.shape[0] == 4


def test_build_from_splits_missing_split_raises():
    """Missing required split key in x_splits or y_splits must raise KeyError."""
    x_splits, y_splits, nodes, od_tensor = make_spatio_splits()
    x_splits.pop("train")

    loader_obj = SpatioTorchDataLoader(batch_size=4, device=torch.device("cpu"))

    with pytest.raises(KeyError, match="Missing split 'train'"):
        loader_obj.build_from_splits((x_splits, y_splits, nodes, od_tensor))


def test_get_adj_matrix_returns_none_if_not_set():
    """get_adj_matrix must return None before any build_from_splits call."""
    loader_obj = SpatioTorchDataLoader(batch_size=2, device=torch.device("cpu"))
    assert loader_obj.get_adj_matrix() is None
