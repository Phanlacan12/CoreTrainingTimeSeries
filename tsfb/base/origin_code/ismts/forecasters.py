"""Turn the decay encoders into multi-step forecasters.

This is the only place where a classifier becomes a forecaster. The source models
(GRU-D, BRITS, T-LSTM as published, and as reproduced in the ISMTS-Review harness) emit
classification logits; here the encoder is kept intact and only the head is replaced.

Two design decisions worth stating, because both had a plausible alternative:

* **Direct multi-step, not autoregressive rollout.** A rollout would need future
  ``delta`` and ``mask``, which do not exist at forecast time. Direct multi-step also
  matches every other deep model in tsfb, so the ``y`` produced by the window builder is
  already the right shape.
* **The recurrence is not touched.** The decay mechanism is the entire reason for the
  port, and it is the part that carries the numerical-equivalence tests.

Models take the MASKED_DECAY batch and return ``(B, pred_len, c_out)``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tsfb.base.origin_code.ismts.encoders import (
    BRITSEncoder,
    GRUDEncoder,
    TLSTMEncoder,
)


class ForecastHead(nn.Module):
    """Final hidden state -> (pred_len, n_out), matching the paper's FC-128/FC-64 head."""

    def __init__(self, hidden: int, pred_len: int, n_out: int, p_drop: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * n_out),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(-1, self.pred_len, self.n_out)


def _dims(config: SimpleNamespace) -> tuple[int, int, int, int]:
    d_in = int(getattr(config, "enc_in", getattr(config, "c_out", 1)))
    hidden = int(getattr(config, "d_model", 64))
    pred_len = int(getattr(config, "pred_len", getattr(config, "horizon", 1)))
    c_out = int(getattr(config, "c_out", d_in))
    return d_in, hidden, pred_len, c_out


class GRUDForecaster(nn.Module):
    """GRU-D encoder + direct multi-step head."""

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        d_in, hidden, pred_len, c_out = _dims(config)
        self.encoder = GRUDEncoder(d_in, hidden)
        self.head = ForecastHead(hidden, pred_len, c_out)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        x_last: torch.Tensor,
    ) -> torch.Tensor:
        states, _ = self.encoder(x, m, d, x_last)
        return self.head(states[:, -1])


class TLSTMForecaster(nn.Module):
    """T-LSTM encoder + direct multi-step head.

    Consumes ``x_last`` as the dense (forward-filled) input, since T-LSTM has no
    imputation mechanism of its own -- its time-awareness lives entirely in the cell
    decay.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        d_in, hidden, pred_len, c_out = _dims(config)
        self.encoder = TLSTMEncoder(d_in, hidden)
        self.head = ForecastHead(hidden, pred_len, c_out)
        self.delta_scale = float(getattr(config, "delta_scale", 1.0))

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        x_last: torch.Tensor,
    ) -> torch.Tensor:
        # Per-step elapsed time: the first feature column of `d` carries the step gap
        # for any feature observed at the previous step, so take the row minimum as the
        # sequence-level dt and undo the scaling the window builder applied.
        dt = d.min(dim=-1).values * self.delta_scale
        x_fill = m * x + (1.0 - m) * x_last
        states = self.encoder(x_fill, m, dt)
        return self.head(states[:, -1])


class BRITSForecaster(nn.Module):
    """Bidirectional BRITS encoder + direct multi-step head.

    Note the encoder is bidirectional over the LOOKBACK WINDOW only; it never sees
    anything past the forecast origin, so this stays causal at the window level.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        d_in, hidden, pred_len, c_out = _dims(config)
        self.encoder = BRITSEncoder(d_in, hidden)
        self.head = ForecastHead(2 * hidden, pred_len, c_out)
        self.delta_scale = float(getattr(config, "delta_scale", 1.0))
        self.consistency_weight = float(getattr(config, "consistency_weight", 0.1))
        self.last_consistency: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        x_last: torch.Tensor,
    ) -> torch.Tensor:
        dt = d.min(dim=-1).values * self.delta_scale
        h, _, consistency = self.encoder(x, m, d, dt)
        # Exposed so the trainer can add it if it wants; kept off the returned tensor so
        # the model still satisfies the plain (B, pred_len, c_out) contract.
        self.last_consistency = consistency
        return self.head(h)
