# ISMTS vendored code

Decay-based encoders for irregularly sampled time series, adapted from the
ISMTS-Review benchmark harness — a PyTorch reproduction of

> Sun, Hong, Song & Li. *A Review of Deep Learning Methods for Irregularly Sampled
> Medical Time Series Data.* arXiv:2010.12493

| | |
|---|---|
| Source repo | `core_timeseries_forecasting/survey_irregular_0/ISMTS-Review` |
| Source path | `benchmark/models/{grud,brits,tlstm}.py`, `benchmark/models/base.py` |
| Source commit | `cef6cb5` |

## What was changed on the way in

* **Input signature.** The source models take a `Batch` dataclass that also carries
  classification-fold machinery (labels, elimination masks, cached reversals). Here the
  encoders take plain tensors `(x, m, d, x_last)` — the MASKED_DECAY contract.
* **Head.** The source emits classification logits. `forecasters.py` drops that head and
  attaches a direct multi-step `ForecastHead`; the recurrences are untouched.
* **BRITS reversal.** The source precomputes the time-reversed view once per CV fold in
  `FoldTensors`. There is no fold here, so `BRITSEncoder._reverse_delta` recomputes it
  per batch from `m` and `dt`.

## What must NOT drift

The per-step loops were optimised in the source repo: loop-invariant projections hoisted
out of the recurrence, gate matrices fused. That rewrite is locked there by tests
asserting the vectorised form is numerically identical to the naive loop, and the same
tests are ported to `tests/test_module/test_models/test_ismts_encoders.py`. If you edit a
recurrence here, run those tests.

`derive_temporal` — which guarantees `last_observed` is strictly causal — deliberately
does **not** live in this folder. It is a correctness contract, so it sits in
`tsfb/base/data_module/preprocess_model/components/irregular_tensors.py` where the
linters and the test suite reach it.
