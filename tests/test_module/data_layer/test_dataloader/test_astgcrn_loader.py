import numpy as np
import pytest
import torch

from tsfb.base.data_module.dataloader.astgcrn_loader import ASTGCRNDataLoader


@pytest.fixture
def basic_splits() -> tuple[dict, dict, list[str], np.ndarray]:
    n_nodes = 3
    feat_dim = 2
    t_in = 4
    t_out = 5

    train_x = np.random.randn(6, n_nodes, feat_dim, t_in).astype(np.float32)
    val_x = np.random.randn(2, n_nodes, feat_dim, t_in).astype(np.float32)
    test_x = np.random.randn(2, n_nodes, feat_dim, t_in).astype(np.float32)

    train_y = np.random.randn(6, n_nodes, t_out).astype(np.float32)
    val_y = np.random.randn(2, n_nodes, t_out).astype(np.float32)
    test_y = np.random.randn(2, n_nodes, t_out).astype(np.float32)

    x_splits = {"train": train_x, "val": val_x, "test": test_x}
    y_splits = {"train": train_y, "val": val_y, "test": test_y}

    nodes = [f"n{i}" for i in range(n_nodes)]
    od_tensor = np.ones((n_nodes, n_nodes), dtype=np.float32)

    return x_splits, y_splits, nodes, od_tensor


def test_build_from_splits_basic(basic_splits) -> None:
    """build_from_splits creates loaders and sets metadata correctly."""
    x_splits, y_splits, nodes, od_tensor = basic_splits

    mean = np.zeros((1, 1, 1, 1), dtype=np.float32)
    std = np.ones((1, 1, 1, 1), dtype=np.float32)

    dl = ASTGCRNDataLoader(batch_size=4, device=torch.device("cpu"))

    train_loader, val_loader, test_loader = dl.build_from_splits(
        (x_splits, y_splits, nodes, od_tensor),
        mean=mean,
        std=std,
    )

    assert len(train_loader.dataset) == 6
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 2

    x0, y0 = train_loader.dataset[0]
    assert x0.shape == (3, 2, 4)
    assert y0.shape == (3, 5)

    assert dl.nodes == nodes
    assert dl.get_adj_matrix() is od_tensor
    assert dl.n_nodes == 3
    assert dl.feat_dim == 2
    assert dl.t_in == 4
    assert np.array_equal(dl.mean, mean)
    assert np.array_equal(dl.std, std)

    assert dl.val_target_tensor is not None
    assert dl.val_target_tensor.shape == (2, 3, 5)
    assert dl.val_target_tensor.dtype == torch.float32


def test_build_from_splits_empty_train() -> None:
    """Empty train split still produces loaders; shape metadata stays None."""
    n_nodes = 2
    feat_dim = 1
    t_in = 3
    t_out = 2

    train_x = np.empty((0, n_nodes, feat_dim, t_in), dtype=np.float32)
    val_x = np.random.randn(2, n_nodes, feat_dim, t_in).astype(np.float32)
    test_x = np.random.randn(2, n_nodes, feat_dim, t_in).astype(np.float32)

    train_y = np.empty((0, n_nodes, t_out), dtype=np.float32)
    val_y = np.random.randn(2, n_nodes, t_out).astype(np.float32)
    test_y = np.random.randn(2, n_nodes, t_out).astype(np.float32)

    x_splits = {"train": train_x, "val": val_x, "test": test_x}
    y_splits = {"train": train_y, "val": val_y, "test": test_y}
    nodes = [f"n{i}" for i in range(n_nodes)]
    od_tensor = np.eye(n_nodes, dtype=np.float32)

    dl = ASTGCRNDataLoader(batch_size=4, device=torch.device("cpu"))
    train_loader, val_loader, test_loader = dl.build_from_splits(
        (x_splits, y_splits, nodes, od_tensor)
    )

    assert len(train_loader.dataset) == 0
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 2

    assert dl.n_nodes is None
    assert dl.feat_dim is None
    assert dl.t_in is None

    assert dl.val_target_tensor is not None
    assert dl.val_target_tensor.shape == (2, n_nodes, t_out)


def test_get_adj_matrix_returns_stored_tensor(basic_splits) -> None:
    """get_adj_matrix returns the stored adjacency tensor."""
    x_splits, y_splits, nodes, od_tensor = basic_splits
    dl = ASTGCRNDataLoader()

    dl.build_from_splits((x_splits, y_splits, nodes, od_tensor))
    returned = dl.get_adj_matrix()

    assert returned is od_tensor or np.array_equal(returned, od_tensor)
