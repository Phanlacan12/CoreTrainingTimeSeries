import numpy as np
import pandas as pd
import pytest

from tsfb.base.data_module.preprocess_model.astgcrn_preprocess import ASTGCRNPreprocess


@pytest.fixture
def sample_config() -> dict:
    return {
        "num_of_hours": 1,
        "num_of_days": 0,
        "num_of_weeks": 0,
        "num_for_predict": 3,
        "points_per_hour": 1,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "normalize": True,
    }


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Create a simple wide_df with T=20, N=2, F=1."""
    T = 20
    N = 2
    F = 1
    data = np.arange(T * N * F, dtype=np.float32).reshape(T, N * F)
    wide_df = pd.DataFrame(data)
    nodes = [f"n{i}" for i in range(N)]
    od_tensor = np.ones((N, N), dtype=np.float32)
    return wide_df, nodes, od_tensor


def test_fit_and_predict_basic_shapes(sample_config, sample_data) -> None:
    """fit() and predict() should produce consistent splits and shapes."""
    wide_df, nodes, od_tensor = sample_data
    pre = ASTGCRNPreprocess(sample_config)

    x_splits, y_splits, out_nodes, out_od = pre.fit((wide_df, nodes, od_tensor))

    # metadata
    assert out_nodes == nodes
    assert out_od is od_tensor
    for d in (x_splits, y_splits):
        assert set(d.keys()) == {"train", "val", "test"}

    # expected sequence counts:
    # T=20, t_in=1, T_out=3 -> S=17
    # splits: int(17*0.6)=10, int(17*0.2)=3, rest=4
    train_x, val_x, test_x = (
        x_splits["train"],
        x_splits["val"],
        x_splits["test"],
    )
    train_y, val_y, test_y = (
        y_splits["train"],
        y_splits["val"],
        y_splits["test"],
    )

    assert train_x.shape[0] == 10
    assert val_x.shape[0] == 3
    assert test_x.shape[0] == 4

    # X: (S, N, F, t_in) with N=2, F=1, t_in=1
    assert train_x.shape[1:] == (2, 1, 1)
    assert val_x.shape[1:] == (2, 1, 1)
    assert test_x.shape[1:] == (2, 1, 1)

    # Y: (S, N, T_out) with T_out=3
    assert train_y.shape[1:] == (2, 3)
    assert val_y.shape[1:] == (2, 3)
    assert test_y.shape[1:] == (2, 3)

    # normalization stats
    assert pre._mean is not None
    assert pre._std is not None
    assert pre._mean.shape == (1, 1, 1, 1)
    assert pre._std.shape == (1, 1, 1, 1)

    # train_x (already normalized) should have mean ~0 and std > 0
    m = train_x.mean()
    s = train_x.std()
    assert abs(m) < 1e-5
    assert s > 0

    # predict() should concatenate all splits
    x_full, y_full, p_nodes, p_od = pre.predict((wide_df, nodes, od_tensor))
    assert p_nodes == nodes
    assert p_od is od_tensor

    x_concat = np.concatenate(
        [x_splits["train"], x_splits["val"], x_splits["test"]], axis=0
    )
    y_concat = np.concatenate(
        [y_splits["train"], y_splits["val"], y_splits["test"]], axis=0
    )
    assert x_full.shape == x_concat.shape
    assert y_full.shape == y_concat.shape
    assert np.allclose(x_full, x_concat)
    assert np.allclose(y_full, y_concat)


def test_predict_without_fit_raises(sample_config, sample_data) -> None:
    """If normalize=True and fit() not called, predict() must raise RuntimeError."""
    wide_df, nodes, od_tensor = sample_data
    pre = ASTGCRNPreprocess(sample_config)

    with pytest.raises(RuntimeError):
        pre.predict((wide_df, nodes, od_tensor))


def test_no_normalization_identity(sample_data) -> None:
    """With normalize=False, X is not changed and inverse_transform is identity."""
    wide_df, nodes, od_tensor = sample_data
    config = {
        "num_of_hours": 1,
        "num_of_days": 0,
        "num_of_weeks": 0,
        "num_for_predict": 3,
        "points_per_hour": 1,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "normalize": False,
    }
    pre = ASTGCRNPreprocess(config)

    x_splits, _, _, _ = pre.fit((wide_df, nodes, od_tensor))
    assert pre.train_x is not None
    assert np.allclose(x_splits["train"], pre.train_x)

    x_rec = pre.inverse_transform(x_splits["train"])
    assert np.allclose(x_rec, pre.train_x)

    # calling inverse_transform before fit should fail
    pre2 = ASTGCRNPreprocess(config)
    with pytest.raises(RuntimeError):
        pre2.inverse_transform(np.zeros((1, 1, 1, 1), dtype=np.float32))


def test_empty_sequences_fit_and_predict() -> None:
    """When data is too short, fit() and predict() must return empty sequences."""
    config = {
        "num_of_hours": 4,
        "num_of_days": 0,
        "num_of_weeks": 0,
        "num_for_predict": 3,
        "points_per_hour": 1,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "normalize": True,
    }
    T = 5  # t_in=4, T_out=3 -> no sequences
    N = 1
    F = 1
    data = np.arange(T * N * F, dtype=np.float32).reshape(T, N * F)
    wide_df = pd.DataFrame(data)
    nodes = ["n0"]
    od_tensor = np.ones((N, N), dtype=np.float32)

    pre = ASTGCRNPreprocess(config)
    x_splits, y_splits, out_nodes, out_od = pre.fit((wide_df, nodes, od_tensor))

    for key in ("train", "val", "test"):
        assert x_splits[key].shape[0] == 0
        assert y_splits[key].shape[0] == 0

    assert out_nodes == nodes
    assert out_od is od_tensor

    x_full, y_full, p_nodes, p_od = pre.predict((wide_df, nodes, od_tensor))
    assert x_full.shape[0] == 0
    assert y_full.shape[0] == 0
    assert p_nodes == nodes
    assert p_od is od_tensor
