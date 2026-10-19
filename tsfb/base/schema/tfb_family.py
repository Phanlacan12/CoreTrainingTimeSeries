from __future__ import annotations

from enum import StrEnum


class ModelFamily(StrEnum):
    """Supported forward-signature families for integrated tfb models."""

    SEQ2SEQ = "seq2seq_with_time_mark"
    SINGLE_INPUT = "single_input_tensor"
    MASKED_DECAY = "masked_decay_tensors"
    """(x, m, d, x_last) for the irregular decay models: GRU-D, BRITS, T-LSTM."""
