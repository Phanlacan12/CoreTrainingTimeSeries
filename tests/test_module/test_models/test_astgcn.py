import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsfb.base.models.astgcn.astgcn_adapter import ASTGCNAdapter


class _DummyNet(torch.nn.Module):
    """
    Minimal test net:
      - Purpose: fast, deterministic, isolates adapter logic from real ASTGCN.
      - Predictable: easy to compute exact expected outputs.
      - Has a trainable param to satisfy the optimizer.
    I/O: x (B,N,F,T_in) -> y (B,N,T_out) = mean_F(x[..., -1]) repeated T_out.
    """

    def __init__(self, t_out: int):
        super().__init__()
        self.t_out = t_out
        self.scale = torch.nn.Parameter(torch.tensor(1.0))  # satisfy optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,N,F,T_in)
        y = x[..., -1].mean(dim=2).unsqueeze(-1).repeat(1, 1, self.t_out)  # (B,N,T_out)
        return y * self.scale


def _dummy_make_model(device, *args, **kwargs):
    t_out = kwargs.get("num_for_predict", args[7])  # adapter passes positionally
    return _DummyNet(t_out).to(device)


# --- Fixtures ---
@pytest.fixture
def sizes():
    return {"B": 2, "N": 3, "F": 2, "T_in": 6, "T_out": 4}


@pytest.fixture
def config(sizes, tmp_path):
    return {
        "use_cuda": False,
        "num_for_predict": sizes["T_out"],
        "len_input": sizes["T_in"],
        "nb_block": 1,
        "in_channels": sizes["F"],
        "K": 2,
        "nb_chev_filter": 8,
        "nb_time_filter": 8,
        "time_strides": 1,
        "learning_rate": 1e-2,
        "epochs": 1,
        "start_epoch": 0,
        "batch_size": sizes["B"],
        "loss_function": "mae",
        "missing_value": 0.0,
        "print_every": 10_000,
        "val_print_every": 10_000,
        "save_best": False,
        "params_path": str(tmp_path / "ckpts"),
    }


@pytest.fixture
def toy_tuple_train_data(sizes):
    B, N, F, T_in, T_out = (
        sizes["B"],
        sizes["N"],
        sizes["F"],
        sizes["T_in"],
        sizes["T_out"],
    )
    rng = np.random.default_rng(123)
    x = rng.normal(size=(B, N, F, T_in)).astype(np.float32)
    y = np.repeat(x[..., -1].mean(axis=2)[..., None], T_out, axis=2).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    ld = DataLoader(ds, batch_size=B, shuffle=False)
    return (
        ld,
        ld,
        ld,
        np.array(0.0, np.float32),
        np.array(1.0, np.float32),
        N,
        F,
        T_in,
        np.zeros((N, N), np.float32),
    )


# --- Tests ---
def test_forecast_fit_and_forecast(monkeypatch, config, toy_tuple_train_data, sizes):
    import tsfb.base.models.astgcn.astgcn_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "make_model", _dummy_make_model, raising=True)

    model = ASTGCNAdapter(config).forecast_fit(toy_tuple_train_data)

    rng = np.random.default_rng(42)
    N, F, T_in, T_out = sizes["N"], sizes["F"], sizes["T_in"], sizes["T_out"]
    x = rng.normal(size=(N, F, T_in)).astype(np.float32)
    expected = np.repeat(x[:, :, -1].mean(axis=1)[:, None], T_out, axis=1).astype(
        np.float32
    )

    y = model.forecast(horizon=T_out, series=x)
    assert y.shape == (N, T_out)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError):
        model.forecast(horizon=T_out + 1, series=x)


def test_batch_forecast_two_layouts(monkeypatch, config, toy_tuple_train_data, sizes):
    import tsfb.base.models.astgcn.astgcn_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "make_model", _dummy_make_model, raising=True)
    model = ASTGCNAdapter(config).forecast_fit(toy_tuple_train_data)

    B, N, F, T_in, T_out = (
        sizes["B"],
        sizes["N"],
        sizes["F"],
        sizes["T_in"],
        sizes["T_out"],
    )
    rng = np.random.default_rng(7)
    arr_bnft = rng.normal(size=(B, N, F, T_in)).astype(np.float32)
    expected = np.repeat(
        arr_bnft[..., -1].mean(axis=2)[..., None], T_out, axis=2
    ).astype(np.float32)

    # BNFT layout
    class BM1:
        def __init__(self, a):
            self.a = a

        def make_batch(self, b, w):
            return {"series": torch.from_numpy(self.a)}

    out1 = model.batch_forecast(
        horizon=T_out, batch_maker=BM1(arr_bnft), batch_size=B, win_size=T_in
    )
    assert out1.shape == (B, N, T_out)
    np.testing.assert_allclose(out1, expected, rtol=1e-5, atol=1e-5)

    # BTNF layout
    class BM2:
        def __init__(self, a):
            self.a = a

        def make_batch(self, b, w):
            return {"series": torch.from_numpy(self.a)}

    arr_btnf = np.transpose(arr_bnft, (0, 3, 1, 2))
    out2 = model.batch_forecast(
        horizon=T_out, batch_maker=BM2(arr_btnf), batch_size=B, win_size=T_in
    )
    assert out2.shape == (B, N, T_out)
    np.testing.assert_allclose(out2, expected, rtol=1e-5, atol=1e-5)
