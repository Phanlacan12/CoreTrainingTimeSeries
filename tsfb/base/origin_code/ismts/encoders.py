"""Decay-based encoders for irregularly sampled series.

Each encoder consumes the MASKED_DECAY tensor contract and returns the hidden-state
sequence; the forecasting head lives in ``forecasters.py``.

    x       (B, T, D)  values, 0 where unobserved
    m       (B, T, D)  1 where observed
    d       (B, T, D)  time since that feature was last observed, scaled
    x_last  (B, T, D)  last observed value, STRICTLY before the current step

The recurrences are ported from the ISMTS-Review harness with the loop-invariant
projections already hoisted out of the per-step loop and the gate matrices fused --
that rewrite is locked by numerical-equivalence tests in both repos. The only
adaptation is the input signature: the source takes its own ``Batch`` dataclass, which
also carries classification-fold machinery that has no meaning here.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsfb.base.origin_code.ismts.layers import (
    FeatureRegression,
    FilterLinear,
    TemporalDecay,
)


class GRUDEncoder(nn.Module):
    """GRU-D (Che et al., Scientific Reports 2018).

    gamma_x = exp(-relu(W_x . d))   W_x constrained DIAGONAL
    gamma_h = exp(-relu(W_h . d))   full matrix
    x_t    <- m*x + (1-m)*(gamma_x * x_last + (1-gamma_x) * x_mean)
    h_{t-1}<- gamma_h * h_{t-1}
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        # x_mean is 0 because tsfb's preprocess already z-scores the frame.
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.gamma_x = FilterLinear(d_in, d_in, torch.eye(d_in))
        self.gamma_h = nn.Linear(d_in, hidden)
        # z, r and h~ all read [x, m, h]. One fused Linear over 3*hidden for the input
        # half (hoistable out of the loop) and two for the hidden half -- h~ reads r*h
        # while z and r read h, and keeping them separate avoids computing a third of
        # each matmul only to discard it.
        self.gates_in = nn.Linear(2 * d_in, 3 * hidden)
        self.gates_h_zr = nn.Linear(hidden, 2 * hidden, bias=False)
        self.gates_h_c = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        x_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (states (B,T,H), decay_estimate (B,T,D))."""
        batch, steps, _ = x.shape

        # Independent of the hidden state -> computed for all T steps at once.
        gx = torch.exp(-torch.relu(self.gamma_x(d)))
        gh = torch.exp(-torch.relu(self.gamma_h(d)))
        x_hat = gx * x_last + (1.0 - gx) * self.x_mean
        x_in = m * x + (1.0 - m) * x_hat
        gates_in = self.gates_in(torch.cat((x_in, m), dim=-1))

        h = x.new_zeros(batch, self.hidden)
        states = []
        for step in range(steps):
            h = gh[:, step] * h
            gi_z, gi_r, gi_c = gates_in[:, step].chunk(3, dim=-1)
            gh_z, gh_r = self.gates_h_zr(h).chunk(2, dim=-1)
            z = torch.sigmoid(gi_z + gh_z)
            r = torch.sigmoid(gi_r + gh_r)
            h_tilde = torch.tanh(gi_c + self.gates_h_c(r * h))
            h = (1.0 - z) * h + z * h_tilde
            states.append(h)

        return torch.stack(states, dim=1), x_hat


class TLSTMEncoder(nn.Module):
    """T-LSTM (Baytas et al., KDD 2017).

    Discounts only the SHORT-term memory by the elapsed time:

        C_short  = tanh(W_d C_{t-1} + b_d)
        C*_{t-1} = (C_{t-1} - C_short) + C_short * g(dt),  g(dt) = 1 / log(e + dt)

    ``dt`` is in RAW time units -- the decay is logarithmic, so unlike GRU-D's
    exp(-relu(.)) it does not saturate on that scale.
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.decomp = nn.Linear(hidden, hidden)
        self.gates_in = nn.Linear(2 * d_in, 4 * hidden)
        self.gates_h = nn.Linear(hidden, 4 * hidden, bias=False)

    @staticmethod
    def _g(dt: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.log(math.e + dt)

    def forward(
        self,
        x_fill: torch.Tensor,
        m: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """``dt`` is (B, T) elapsed time between consecutive steps. Returns (B,T,H)."""
        batch, steps, _ = x_fill.shape
        h = x_fill.new_zeros(batch, self.hidden)
        c = x_fill.new_zeros(batch, self.hidden)

        gates_in = self.gates_in(torch.cat([x_fill, m], dim=-1))
        g_dt = self._g(dt).unsqueeze(-1)

        states = []
        for step in range(steps):
            c_short = torch.tanh(self.decomp(c))
            c = (c - c_short) + c_short * g_dt[:, step]

            i, f, o, c_tilde = (gates_in[:, step] + self.gates_h(h)).chunk(4, dim=-1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            c = f * c + i * torch.tanh(c_tilde)
            h = o * torch.tanh(c)
            states.append(h)

        return torch.stack(states, dim=1)


class RITSEncoder(nn.Module):
    """One direction of BRITS (Cao et al., NeurIPS 2018)."""

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.rnn_cell = nn.LSTMCell(d_in * 2, hidden)
        self.temp_decay_h = TemporalDecay(d_in, hidden, diag=False)
        self.temp_decay_x = TemporalDecay(d_in, d_in, diag=True)
        self.hist_reg = nn.Linear(hidden, d_in)
        self.feat_reg = FeatureRegression(d_in)
        self.weight_combine = nn.Linear(d_in * 2, d_in)

    def forward(
        self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (final hidden (B,H), completed series (B,T,D))."""
        batch, steps, _ = x.shape

        # Read only d and m -> hoisted out of the loop.
        gamma_h = self.temp_decay_h(d)
        gamma_x = self.temp_decay_x(d)
        alpha = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=-1)))

        h = x.new_zeros(batch, self.hidden)
        c = x.new_zeros(batch, self.hidden)
        imputations = []

        for step in range(steps):
            x_t, m_t = x[:, step], m[:, step]
            h = h * gamma_h[:, step]

            x_h = self.hist_reg(h)
            x_c = m_t * x_t + (1 - m_t) * x_h
            z_h = self.feat_reg(x_c)
            c_h = alpha[:, step] * z_h + (1 - alpha[:, step]) * x_h
            c_c = m_t * x_t + (1 - m_t) * c_h

            h, c = self.rnn_cell(torch.cat([c_c, m_t], dim=1), (h, c))
            imputations.append(c_c)

        return h, torch.stack(imputations, dim=1)


class BRITSEncoder(nn.Module):
    """Bidirectional BRITS: forward and backward RITS, concatenated.

    The backward direction consumes the time-reversed sequence with ``d`` recomputed in
    that direction -- flipping ``d`` would attribute each gap to the wrong side of the
    step.
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.fwd = RITSEncoder(d_in, hidden)
        self.bwd = RITSEncoder(d_in, hidden)

    @staticmethod
    def _reverse_delta(m: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Recompute per-feature gaps for the reversed sequence."""
        m_r = torch.flip(m, dims=[1])
        gaps_r = torch.flip(dt, dims=[1]).roll(shifts=-1, dims=1)
        gaps_r[:, -1] = 0.0
        d_r = torch.zeros_like(m_r)
        for step in range(1, d_r.shape[1]):
            d_r[:, step] = (
                gaps_r[:, step, None] + (1 - m_r[:, step - 1]) * d_r[:, step - 1]
            )
        return d_r

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (concatenated hidden (B,2H), imputation (B,T,D), consistency loss)."""
        h_f, imp_f = self.fwd(x, m, d)

        x_r = torch.flip(x, dims=[1])
        m_r = torch.flip(m, dims=[1])
        d_r = self._reverse_delta(m, dt)
        h_b, imp_b_rev = self.bwd(x_r, m_r, d_r)
        imp_b = torch.flip(imp_b_rev, dims=[1])

        consistency = (imp_f - imp_b).abs().mean()
        return torch.cat([h_f, h_b], dim=-1), (imp_f + imp_b) / 2.0, consistency
