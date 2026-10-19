"""Tests for the irregular-series (MASKED_DECAY) path.

Three things are pinned here, in increasing order of how easy they are to lose:

1. `derive_temporal`'s strict causality -- the guarantee that makes a decay estimate
   scoreable against the value it is estimating.
2. That the vectorised encoders are numerically identical to a naive per-step loop over
   the same parameters. The source repo hoisted loop-invariant projections and fused the
   gate matrices for a 12-24x speedup; these tests are what stop that optimisation from
   silently changing the model.
3. That causality survives *windowing*, which is where it can actually be lost: slicing
   a causally-derived array is only safe if nothing downstream looks forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tsfb.base.data_module.preprocess_model.components.irregular_tensors import (
    derive_temporal,
    forward_fill,
    observation_mask_from_frame,
)
from tsfb.base.data_module.preprocess_model.components.window_builder import (
    build_inference_samples,
    build_training_samples,
)
from tsfb.base.origin_code.ismts.encoders import BRITSEncoder, GRUDEncoder, TLSTMEncoder
from tsfb.base.origin_code.ismts.forecasters import (
    BRITSForecaster,
    GRUDForecaster,
    TLSTMForecaster,
)
from tsfb.base.schema.tfb_family import ModelFamily


# --------------------------------------------------------------------- temporal maths
def test_derive_temporal_hand_example():
    #                t=0    t=1    t=2    t=3
    mask = np.array([[[1], [0], [0], [1]]], dtype=np.uint8)
    values = np.array([[[5.0], [0.0], [0.0], [9.0]]], dtype=np.float32)
    times = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)

    out = derive_temporal(mask, times, values)

    assert out.delta[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    # strictly causal: 0 before the first obs, and never the value at the current step
    assert out.last_observed[0, :, 0].tolist() == [0.0, 5.0, 5.0, 5.0]
    assert out.last_valid[0, :, 0].tolist() == [0, 1, 1, 1]


def test_last_observed_never_reveals_current_value():
    """If last_observed[t] contained x[t], any decay model could score a perfect
    imputation loss at observed entries by driving gamma -> 1."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 12, 3)).astype(np.float32)
    mask = (rng.random((4, 12, 3)) > 0.4).astype(np.uint8)
    values = values * mask
    times = np.tile(np.arange(12, dtype=np.float32), (4, 1))

    out = derive_temporal(mask, times, values)
    observed = mask.astype(bool)
    coincide = np.isclose(out.last_observed[observed], values[observed]) & (
        values[observed] != 0
    )
    assert coincide.mean() < 0.05


def test_derive_temporal_is_shift_invariant():
    rng = np.random.default_rng(1)
    mask = (rng.random((2, 10, 3)) > 0.5).astype(np.uint8)
    values = (rng.normal(size=(2, 10, 3)) * mask).astype(np.float32)
    times = np.tile(np.arange(10, dtype=np.float32), (2, 1))

    base = derive_temporal(mask, times, values)
    shifted = derive_temporal(mask, times + 137.0, values)
    assert np.allclose(base.delta[:, 1:], shifted.delta[:, 1:])


def test_forward_fill_carries_last_observation():
    values = np.array([[[3.0], [0.0], [7.0], [0.0]]], dtype=np.float32)
    mask = np.array([[[1], [0], [1], [0]]], dtype=np.uint8)
    assert forward_fill(values, mask).ravel().tolist() == [3.0, 3.0, 7.0, 7.0]


def test_observation_mask_prefers_imputed_flags():
    """After upstream imputation the frame has no NaN, so notna() would be all ones and
    the decay mechanism would see a perfectly regular series."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    flags = pd.DataFrame({"a": [False, True, False]})
    assert observation_mask_from_frame(frame, ["a"], flags).ravel().tolist() == [
        1,
        0,
        1,
    ]
    assert observation_mask_from_frame(frame, ["a"]).ravel().tolist() == [1, 1, 1]


# ------------------------------------------------ vectorised encoders == naive loops
def _toy(batch=3, steps=16, dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(batch, steps, dim, generator=g) > 0.4).float()
    x = torch.randn(batch, steps, dim, generator=g) * m
    d = torch.rand(batch, steps, dim, generator=g)
    x_last = torch.randn(batch, steps, dim, generator=g)
    return x, m, d, x_last


def test_grud_encoder_matches_naive_loop():
    x, m, d, x_last = _toy()
    torch.manual_seed(0)
    enc = GRUDEncoder(x.shape[2], 32).eval()

    with torch.no_grad():
        states, _ = enc(x, m, d, x_last)
        h = torch.zeros(x.shape[0], 32)
        ref = []
        for i in range(x.shape[1]):
            gx_i = torch.exp(-torch.relu(enc.gamma_x(d[:, i])))
            gh_i = torch.exp(-torch.relu(enc.gamma_h(d[:, i])))
            x_hat = gx_i * x_last[:, i] + (1 - gx_i) * enc.x_mean
            x_in = m[:, i] * x[:, i] + (1 - m[:, i]) * x_hat
            h = gh_i * h
            gi_z, gi_r, gi_c = enc.gates_in(torch.cat([x_in, m[:, i]], dim=-1)).chunk(
                3, dim=-1
            )
            gh_z, gh_r = enc.gates_h_zr(h).chunk(2, dim=-1)
            z = torch.sigmoid(gi_z + gh_z)
            r = torch.sigmoid(gi_r + gh_r)
            h = (1 - z) * h + z * torch.tanh(gi_c + enc.gates_h_c(r * h))
            ref.append(h)
    assert torch.allclose(states, torch.stack(ref, 1), atol=1e-5)


def test_tlstm_encoder_matches_naive_loop():
    x, m, _, _ = _toy()
    dt = torch.rand(x.shape[0], x.shape[1])
    torch.manual_seed(0)
    enc = TLSTMEncoder(x.shape[2], 32).eval()

    with torch.no_grad():
        states = enc(x, m, dt)
        h = torch.zeros(x.shape[0], 32)
        c = torch.zeros(x.shape[0], 32)
        ref = []
        for k in range(x.shape[1]):
            c_short = torch.tanh(enc.decomp(c))
            c = (c - c_short) + c_short * enc._g(dt[:, k]).unsqueeze(-1)
            gate = enc.gates_in(torch.cat([x[:, k], m[:, k]], -1)) + enc.gates_h(h)
            i, f, o, ct = gate.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(ct)
            h = torch.sigmoid(o) * torch.tanh(c)
            ref.append(h)
    assert torch.allclose(states, torch.stack(ref, 1), atol=1e-5)


def test_brits_backward_delta_is_recomputed_not_flipped():
    """Flipping `d` would attribute each gap to the wrong side of the step."""
    x, m, d, _ = _toy()
    dt = torch.ones(x.shape[0], x.shape[1])
    enc = BRITSEncoder(x.shape[2], 16).eval()
    d_rev = enc._reverse_delta(m, dt)
    assert not torch.allclose(d_rev, torch.flip(d, dims=[1]))
    assert torch.all(d_rev[:, 0] == 0)


# ------------------------------------------------------------------ window contract
def _gappy_frame(n=64, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    frame.iloc[5:9, 0] = np.nan
    frame.iloc[20, 1] = np.nan
    return frame


def test_window_builder_emits_masked_decay_keys():
    train, _ = build_training_samples(
        _gappy_frame(),
        family=ModelFamily.MASKED_DECAY,
        value_cols=["a", "b"],
        seq_len=8,
        pred_len=4,
        label_len=4,
        freq="h",
        train_ratio=1.0,
    )
    for key in ("x", "m", "d", "x_last", "y", "y_mask"):
        assert key in train, f"missing {key}"
        assert train[key].ndim == 3
    # the whole point: the mask must not be all ones on a frame with gaps
    assert train["m"].min() == 0.0


def test_masked_decay_window_causality():
    """Mutating rows AFTER a window's forecast origin must not change that window."""
    frame = _gappy_frame()
    kwargs = dict(
        family=ModelFamily.MASKED_DECAY,
        value_cols=["a", "b"],
        seq_len=8,
        pred_len=4,
        label_len=4,
        freq="h",
        train_ratio=1.0,
    )
    before, _ = build_training_samples(frame, **kwargs)

    mutated = frame.copy()
    mutated.iloc[40:] = mutated.iloc[40:] + 100.0
    after, _ = build_training_samples(mutated, **kwargs)

    # windows whose forecast origin sits at or before row 40 must be untouched
    n_safe = 40 - kwargs["seq_len"] - kwargs["pred_len"] + 1
    for key in ("x", "m", "d", "x_last"):
        assert np.allclose(before[key][:n_safe], after[key][:n_safe]), key


def test_inference_delta_carries_pre_window_gap():
    """`d` at the first tail step reflects the gap before the window."""
    frame = _gappy_frame()
    frame.iloc[-10:-6, 0] = np.nan
    samples = build_inference_samples(
        frame,
        family=ModelFamily.MASKED_DECAY,
        value_cols=["a", "b"],
        seq_len=4,
        pred_len=2,
        label_len=2,
        freq="h",
    )
    assert samples["d"].shape == (1, 4, 2)
    assert samples["d"][0, 0, 0] > 0.0


# ----------------------------------------------------------------------- forecasters
@pytest.mark.parametrize("cls", [GRUDForecaster, TLSTMForecaster, BRITSForecaster])
def test_forecaster_output_shape(cls):
    from types import SimpleNamespace

    x, m, d, x_last = _toy(batch=5, steps=12, dim=3)
    cfg = SimpleNamespace(enc_in=3, c_out=3, d_model=16, pred_len=6, delta_scale=48.0)
    model = cls(cfg).eval()
    with torch.no_grad():
        out = model(x, m, d, x_last)
    assert out.shape == (5, 6, 3)
    assert torch.isfinite(out).all()
