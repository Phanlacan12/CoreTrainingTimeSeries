import numpy as np
import pandas as pd
import pytest

from tsfb.base.data_module.preprocess_model.base_preprocess import DefaultPreprocessData


def make_raw_data(n: int = 10):
    """Create a simple dummy dataset for target, past, future and static."""
    idx = pd.date_range("2021-01-01", periods=n, freq="H")
    target = pd.DataFrame({"y": np.arange(n, dtype=float)}, index=idx)
    past = pd.DataFrame({"x1": np.arange(n, dtype=float)}, index=idx)
    future = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=idx)
    static = pd.DataFrame({"s1": np.arange(3, dtype=float)})
    return target, past, future, static


def test_fit_without_normalization_splits_and_scalers():
    """fit() without normalization must split correctly and leave scalers as None."""
    config = {
        "normalize": False,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "data_freq": "H",
    }
    pre = DefaultPreprocessData(config)
    raw = make_raw_data(10)

    splits_tgt, splits_cov, static_out = pre.fit(raw)

    # target splits lengths
    assert len(splits_tgt["train"]) == 6
    assert len(splits_tgt["val"]) == 2
    assert len(splits_tgt["test"]) == 2

    # past split by time, future copied to all splits
    assert len(splits_cov["past_covariates"]["train"]) == 6
    assert len(splits_cov["past_covariates"]["val"]) == 2
    assert len(splits_cov["past_covariates"]["test"]) == 2

    assert len(splits_cov["future_covariates"]["train"]) == 10
    assert len(splits_cov["future_covariates"]["val"]) == 10
    assert len(splits_cov["future_covariates"]["test"]) == 10

    # no normalization -> scalers are None
    assert pre.target_scalers is None
    assert pre.past_scalers is None
    assert pre.future_scalers is None
    assert pre.static_scalers is None

    # static stays as DataFrame with same shape
    assert isinstance(static_out, pd.DataFrame)
    assert list(static_out.columns) == ["s1"]
    assert len(static_out) == 3


def test_fit_with_normalization_fits_scalers_and_scales_train():
    """fit() with normalization must fit scalers and standardize train split."""
    n = 8
    config = {
        "normalize": {"method": "zscore"},
        "split_ratio": {"train": 0.5, "val": 0.25, "test": 0.25},
        "data_freq": "H",
    }
    pre = DefaultPreprocessData(config)
    raw = make_raw_data(n)

    splits_tgt, _, static_out = pre.fit(raw)

    # scalers must exist
    assert pre.target_scalers is not None
    assert pre.past_scalers is not None
    assert pre.future_scalers is not None
    assert pre.static_scalers is not None

    # target train split: mean ≈ 0, std ≈ 1
    train_df = splits_tgt["train"]
    assert np.allclose(train_df.mean().values, 0.0, atol=1e-6)
    assert np.allclose(train_df.std(ddof=0).values, 1.0, atol=1e-6)

    # static scaled to zero mean
    assert isinstance(static_out, pd.DataFrame)
    assert np.allclose(static_out.mean().values, 0.0, atol=1e-6)


def test_predict_with_normalization_requires_fit():
    """predict() with normalize=True must raise if fit() has not been called."""
    config = {
        "normalize": True,
        "data_freq": "H",
    }
    pre = DefaultPreprocessData(config)
    raw = make_raw_data(5)

    with pytest.raises(RuntimeError, match="Scalers are not fitted"):
        pre.predict(raw)


def test_predict_after_fit_preserves_shapes():
    """After fit(), predict() should keep shapes of target, covariates and static."""
    config = {
        "normalize": True,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "data_freq": "H",
    }
    pre = DefaultPreprocessData(config)
    raw = make_raw_data(10)

    pre.fit(raw)
    target_pred, cov_pred, static_pred = pre.predict(raw)

    target_df, past_df, future_df, static_df = raw

    assert isinstance(target_pred, pd.DataFrame)
    assert target_pred.shape == target_df.shape

    assert cov_pred["past_covariates"] is not None
    assert cov_pred["future_covariates"] is not None
    assert cov_pred["past_covariates"].shape == past_df.shape
    assert cov_pred["future_covariates"].shape == future_df.shape

    assert isinstance(static_pred, pd.DataFrame)
    assert static_pred.shape == static_df.shape


def test_split_mode_last():
    """split_mode='last' must allocate tail samples to val/test and the rest to train."""
    raw = make_raw_data(10)
    config = {
        "normalize": False,
        "split_mode": "last",
        "split_last": {"val": 2, "test": 3},  # val=last 2 before test, test=last 3
        "data_freq": "H",
    }
    pre = DefaultPreprocessData(config)
    splits_tgt, splits_cov, static_out = pre.fit(raw)

    # lengths
    assert len(splits_tgt["train"]) == 10 - 2 - 3  # = 5
    assert len(splits_tgt["val"]) == 2
    assert len(splits_tgt["test"]) == 3

    # past split matches target, future copied
    assert len(splits_cov["past_covariates"]["train"]) == 5
    assert len(splits_cov["past_covariates"]["val"]) == 2
    assert len(splits_cov["past_covariates"]["test"]) == 3

    assert len(splits_cov["future_covariates"]["train"]) == 10
    assert len(splits_cov["future_covariates"]["val"]) == 10
    assert len(splits_cov["future_covariates"]["test"]) == 10

    # static unchanged
    assert len(static_out) == 3


def test_split_mode_threshold():
    """split_mode='threshold' must assign segments by time boundaries."""
    raw = make_raw_data(10)
    target_df, past_df, future_df, static_df = raw

    # choose datetime boundaries
    t_train_start = target_df.index[0]
    t_train_end = target_df.index[3]
    t_val_start = target_df.index[4]
    t_val_end = target_df.index[6]

    config = {
        "normalize": False,
        "split_mode": "threshold",
        "split_threshold": {
            "train": [t_train_start, t_train_end],
            "val": [t_val_start, t_val_end],
            # omit test => defaults to > t_val_end
        },
        "data_freq": "H",
    }
    pre = DefaultPreprocessData(config)
    splits_tgt, splits_cov, static_out = pre.fit(raw)

    # compute expected
    idx = target_df.index
    train_mask = (idx >= t_train_start) & (idx <= t_train_end)
    val_mask = (idx >= t_val_start) & (idx <= t_val_end)
    test_mask = idx > t_val_end

    assert len(splits_tgt["train"]) == train_mask.sum()
    assert len(splits_tgt["val"]) == val_mask.sum()
    assert len(splits_tgt["test"]) == test_mask.sum()

    # past split follows target
    assert len(splits_cov["past_covariates"]["train"]) == train_mask.sum()
    assert len(splits_cov["past_covariates"]["val"]) == val_mask.sum()
    assert len(splits_cov["past_covariates"]["test"]) == test_mask.sum()

    # future cov copied
    assert len(splits_cov["future_covariates"]["train"]) == len(target_df)
    assert len(splits_cov["future_covariates"]["val"]) == len(target_df)
    assert len(splits_cov["future_covariates"]["test"]) == len(target_df)

    # static unchanged
    assert len(static_out) == 3
