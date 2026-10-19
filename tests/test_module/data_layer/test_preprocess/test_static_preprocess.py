import numpy as np
import pandas as pd
import pytest

from tsfb.base.data_module.preprocess_model.static_preprocess import (
    StaticGroupedPreprocess,
)


def make_multi_entity_df(
    n_entities: int = 3,
    n_time: int = 10,
) -> pd.DataFrame:
    """Create a simple multi-entity DataFrame with target, covariates and static."""
    rows = []
    dates = pd.date_range("2021-01-01", periods=n_time, freq="D")
    for e in range(n_entities):
        store_id = f"s{e}"
        static_val = 100 + e
        for t_idx, ds in enumerate(dates):
            rows.append(
                {
                    "ds": ds,
                    "store": store_id,
                    "y": float(t_idx),
                    "x1": float(t_idx * 10 + e),
                    "f1": float(t_idx * 100 + e),
                    "s1": float(static_val),
                }
            )
    return pd.DataFrame(rows)


def make_config(
    per_entity: bool = True,
    normalize: bool = True,
    method: str = "zscore",
):
    if normalize:
        normalize_cfg = {"method": method, "per_entity": per_entity}
    else:
        normalize_cfg = {}

    return {
        "group_col": "store",
        "time_col": "ds",
        "target_columns": ["y"],
        "covariate": {
            "past": ["x1"],
            "future": ["f1"],
            "static": ["s1"],
        },
        "normalize": normalize_cfg,
    }


def test_build_grouped_full_basic():
    """_build_grouped_full must keep time/group as columns in grouped long format."""
    df = make_multi_entity_df(n_entities=2, n_time=5)
    cfg = make_config(normalize=False)
    p = StaticGroupedPreprocess(cfg)

    grouped = p._build_grouped_full(df)

    assert p.time_col in grouped.columns
    assert pd.api.types.is_datetime64_any_dtype(grouped[p.time_col])
    for col in ["store", "y", "x1", "f1", "s1"]:
        assert col in grouped.columns
    first_row = grouped.iloc[0]
    assert first_row["store"] == "s0"


def test_fit_per_entity_scaling_stats():
    """Per-entity z-score normalization must standardize y/x1/f1 within each entity."""
    df = make_multi_entity_df(n_entities=2, n_time=10)
    cfg = make_config(per_entity=True, normalize=True, method="zscore")
    p = StaticGroupedPreprocess(cfg)

    out_dict, meta, extra = p.fit(df)

    assert isinstance(out_dict, dict)
    assert "data" in out_dict
    assert meta == {}
    assert extra is None

    df_scaled = out_dict["data"]
    assert p.df_full is df_scaled

    # per-entity scalers present; no global scaler
    assert p.global_scaler is None
    assert set(p.entity_scalers.keys()) == {"s0", "s1"}

    for gid, edf in p.entity_dfs.items():
        cols = ["y", "x1", "f1"]
        edf_feat = edf[cols]
        mean = edf_feat.mean(axis=0)
        std = edf_feat.std(axis=0, ddof=0)
        assert np.allclose(mean.values, 0.0, atol=1e-6)
        if edf_feat.shape[0] > 1:
            assert np.allclose(std.values, 1.0, atol=1e-6)


def test_fit_global_scaling_minmax():
    """Global min-max normalization must scale y/x1/f1 to [0, 1] over all entities."""
    df = make_multi_entity_df(n_entities=2, n_time=10)
    cfg = make_config(per_entity=False, normalize=True, method="minmax")
    p = StaticGroupedPreprocess(cfg)

    out_dict, _, _ = p.fit(df)
    df_scaled = out_dict["data"]

    assert p.global_scaler is not None
    assert p.entity_scalers == {}

    for col in ["y", "x1", "f1"]:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        assert np.isclose(col_min, 0.0, atol=1e-6)
        assert np.isclose(col_max, 1.0, atol=1e-6)


def test_predict_without_normalization_returns_unscaled():
    """When normalize=False, predict() must return the same values as _build_grouped_full."""
    df = make_multi_entity_df()
    cfg = make_config(normalize=False)
    p = StaticGroupedPreprocess(cfg)

    out_dict, _, _ = p.predict(df)
    df_proc = out_dict["data"]

    df_base = p._build_grouped_full(df)

    pd.testing.assert_frame_equal(df_proc.sort_index(), df_base.sort_index())


def test_predict_per_entity_requires_fit():
    """Per-entity normalization must raise if predict() is called before fit()."""
    df = make_multi_entity_df()
    cfg = make_config(per_entity=True, normalize=True, method="zscore")
    p = StaticGroupedPreprocess(cfg)

    with pytest.raises(RuntimeError, match="No per-entity scalers have been fitted"):
        p.predict(df)


def test_predict_per_entity_after_fit_scales_new_data():
    """After fit(), predict() must scale per-entity features differently from raw values."""
    df_train = make_multi_entity_df(n_entities=2, n_time=10)
    cfg = make_config(per_entity=True, normalize=True, method="zscore")
    p = StaticGroupedPreprocess(cfg)
    p.fit(df_train)

    df_new = df_train.copy()
    df_new["y"] = df_new["y"] * 10.0

    df_base_new = p._build_grouped_full(df_new)
    out_dict, _, _ = p.predict(df_new)
    df_proc = out_dict["data"]

    mask_s0 = df_base_new["store"] == "s0"
    y_raw = df_base_new.loc[mask_s0, "y"].values
    y_scaled = df_proc.loc[mask_s0, "y"].values

    assert not np.allclose(y_raw, y_scaled)


def test_build_grouped_full_drops_timezone_from_time_column():
    """Timezone-aware timestamps must be converted to timezone-naive datetimes."""
    df = make_multi_entity_df(n_entities=1, n_time=3)
    df["ds"] = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    cfg = make_config(normalize=False)
    p = StaticGroupedPreprocess(cfg)

    grouped = p._build_grouped_full(df)

    assert str(grouped["ds"].dtype) == "datetime64[ns]"
    assert grouped["ds"].dt.tz is None
