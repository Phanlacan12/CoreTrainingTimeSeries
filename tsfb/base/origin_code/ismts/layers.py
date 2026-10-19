"""Building blocks of the decay-based irregular-series models.

Vendored near-verbatim from the ISMTS-Review benchmark harness so the two copies stay
diffable; see PROVENANCE.md. Each class carries the paper it comes from.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FilterLinear(nn.Module):
    """Linear layer whose weight is masked by a fixed 0/1 matrix.

    With the identity as the mask this gives GRU-D its per-feature *independent* input
    decay: feature j's decay rate may not read feature k's time gap.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        filter_square_matrix: torch.Tensor,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("filter_square_matrix", filter_square_matrix)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.filter_square_matrix * self.weight, self.bias)


class FeatureRegression(nn.Module):
    """Estimate each feature from the OTHER features (zero diagonal).

    BRITS' cross-feature estimate (Cao et al., NeurIPS 2018).
    """

    def __init__(self, size: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(size, size))
        self.b = nn.Parameter(torch.empty(size))
        self.register_buffer("m", torch.ones(size, size) - torch.eye(size))
        stdv = 1.0 / math.sqrt(size)
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W * self.m, self.b)


class TemporalDecay(nn.Module):
    """gamma = exp(-relu(W . delta + b)), optionally with W constrained to a diagonal."""

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = nn.Parameter(torch.empty(output_size, input_size))
        self.b = nn.Parameter(torch.empty(output_size))
        if diag:
            if input_size != output_size:
                raise ValueError("diagonal decay needs input_size == output_size")
            self.register_buffer("m", torch.eye(input_size))
        stdv = 1.0 / math.sqrt(output_size)
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        w = self.W * self.m if self.diag else self.W
        return torch.exp(-F.relu(F.linear(d, w, self.b)))


def masked_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over the entries where ``mask`` is 1."""
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom
