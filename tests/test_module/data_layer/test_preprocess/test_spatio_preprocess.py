import numpy as np
import pandas as pd
import pytest

from tsfb.base.data_module.preprocess_model.spatio_preprocess import SpatioPreprocess


def make_wide_df(
    T: int = 40,
    n_nodes: int = 2,
    n_feat: int = 1,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Create a simple wide_df + nodes list + OD tensor."""
    idx = pd.date_range("2021-01-01", periods=T, freq="H")
    data: dict[str, np.ndarray] = {}
    for i in range(n_nodes):
        for j in range(n_feat):
            col = f"n{i}_f{j}"
            data[col] = np.linspace(0, 1, T, dtype=np.float32) + (i * 10 + j)
    wide_df = pd.DataFrame(data, index=idx)
    nodes = [f"n{i}" for i in range(n_nodes)]
    od_tensor = np.random.rand(n_nodes, n_nodes).astype(np.float32)
    return wide_df, nodes, od_tensor


@pytest.mark.parametrize(
    "normalize_cfg, normalize_enabled, scaler_type",
    [
        ("zscore", True, "zscore"),
        ("minmax", True, "minmax"),
        (False, False, "zscore"),
    ],
)
def test_init_normalize_config(normalize_cfg, normalize_enabled, scaler_type) -> None:
    cfg = {
        "normalize": normalize_cfg,
        "num_for_predict": 4,
        "num_of_hours": 1,
        "points_per_hour": 2,
    }
    p = SpatioPreprocess(cfg)
    assert p.normalize_enabled is normalize_enabled
    assert p.scaler_type == scaler_type


def test_fit_with_normalization_splits_and_shapes() -> None:
    """fit() should build splits, fit scaler and create sequences with correct shape."""
    wide_df, nodes, od = make_wide_df(T=40, n_nodes=2, n_feat=1)

    cfg = {
        "normalize": "zscore",
        "num_for_predict": 4,
        "num_of_hours": 2,
        "points_per_hour": 2,  # num_in = 4
        "split_ratio": {"train": 0.3, "val": 0.35, "test": 0.35},
    }
    p = SpatioPreprocess(cfg)

    x_splits, y_splits, out_nodes, out_od = p.fit((wide_df, nodes, od))

    assert p.scaler is not None
    assert set(x_splits.keys()) == {"train", "val", "test"}
    assert set(y_splits.keys()) == {"train", "val", "test"}
    assert out_nodes == nodes
    assert out_od is od

    n_nodes = len(nodes)
    n_feat = 1
    num_in = cfg["num_of_hours"] * cfg["points_per_hour"]
    num_out = cfg["num_for_predict"]

    train_len = int(len(wide_df) * cfg["split_ratio"]["train"])
    b_train = max(0, train_len - num_in - num_out + 1)

    train_x = x_splits["train"]
    train_y = y_splits["train"]

    if b_train > 0:
        assert train_x.shape == (b_train, n_nodes, n_feat, num_in)
        assert train_y.shape == (b_train, n_nodes, n_feat, num_out)


def test_fit_without_normalization_keeps_raw_values() -> None:
    """When normalize=False, internal arrays should match raw split values."""
    wide_df, nodes, od = make_wide_df(T=20, n_nodes=2, n_feat=1)

    cfg = {
        "normalize": False,
        "num_for_predict": 2,
        "num_of_hours": 1,
        "points_per_hour": 2,
        "split_ratio": {"train": 0.5, "val": 0.25, "test": 0.25},
    }
    p = SpatioPreprocess(cfg)

    x_splits, y_splits, _, _ = p.fit((wide_df, nodes, od))  # noqa: F841

    assert p.scaler is None
    assert p.train_df is not None and p.val_df is not None and p.test_df is not None
    assert np.array_equal(p.train_data, p.train_df.values)
    assert np.array_equal(p.val_data, p.val_df.values)
    assert np.array_equal(p.test_data, p.test_df.values)
    assert set(x_splits.keys()) == {"train", "val", "test"}
    assert set(y_splits.keys()) == {"train", "val", "test"}


def test_predict_requires_fitted_scaler_when_normalize() -> None:
    """predict() should fail if normalization is enabled and fit() was not called."""
    wide_df, nodes, od = make_wide_df(T=20, n_nodes=2, n_feat=1)
    cfg = {
        "normalize": "zscore",
        "num_for_predict": 2,
        "num_of_hours": 1,
        "points_per_hour": 2,
    }
    p = SpatioPreprocess(cfg)

    with pytest.raises(RuntimeError, match="Scaler has not been fit"):
        p.predict((wide_df, nodes, od))


def test_predict_after_fit_shapes_correct() -> None:
    """After fit(), predict() should return full sequences with correct shapes."""
    wide_df, nodes, od = make_wide_df(T=50, n_nodes=2, n_feat=1)
    cfg = {
        "normalize": "zscore",
        "num_for_predict": 3,
        "num_of_hours": 2,
        "points_per_hour": 2,  # num_in = 4
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    p = SpatioPreprocess(cfg)
    p.fit((wide_df, nodes, od))

    x_full, y_full, out_nodes, out_od = p.predict((wide_df, nodes, od))

    num_in = cfg["num_of_hours"] * cfg["points_per_hour"]
    num_out = cfg["num_for_predict"]
    n_nodes = len(nodes)
    n_feat = 1

    b_full = max(0, len(wide_df) - num_in - num_out + 1)

    if b_full > 0:
        assert x_full.shape == (b_full, n_nodes, n_feat, num_in)
        assert y_full.shape == (b_full, n_nodes, n_feat, num_out)

    assert out_nodes == nodes
    assert out_od is od


def test_inverse_transform_roundtrip() -> None:
    """inverse_transform(scaler.transform(X)) should approximately recover X."""
    wide_df, nodes, od = make_wide_df(T=30, n_nodes=2, n_feat=1)
    cfg = {
        "normalize": "zscore",
        "num_for_predict": 2,
        "num_of_hours": 1,
        "points_per_hour": 2,
        "split_ratio": {"train": 0.5, "val": 0.25, "test": 0.25},
    }
    p = SpatioPreprocess(cfg)
    p.fit((wide_df, nodes, od))

    assert p.scaler is not None
    assert p.train_df is not None
    x_sample = p.train_df.values[:5]
    x_scaled = p.scaler.transform(x_sample)
    x_inv = p.inverse_transform(x_scaled)

    assert np.allclose(x_sample, x_inv, atol=1e-6)


def test_get_adj_matrix_returns_od_tensor() -> None:
    """get_adj_matrix() should return the OD tensor passed to fit()."""
    wide_df, nodes, od = make_wide_df(T=10, n_nodes=3, n_feat=1)
    cfg = {
        "normalize": False,
        "num_for_predict": 2,
        "num_of_hours": 1,
        "points_per_hour": 1,
    }
    p = SpatioPreprocess(cfg)
    p.fit((wide_df, nodes, od))

    adj = p.get_adj_matrix()
    assert adj is not None
    assert np.array_equal(adj, od)
