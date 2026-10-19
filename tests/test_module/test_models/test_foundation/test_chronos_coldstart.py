"""Fast contract tests for Chronos multi-series and covariate integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter


class FakeChronosPipeline:
    """Deterministic Chronos-shaped output while recording API arguments."""

    def __init__(self) -> None:
        self.calls = []

    def predict_quantiles(
        self, *, inputs, prediction_length, quantile_levels, **kwargs
    ):
        self.calls.append({"inputs": inputs, **kwargs})
        quantiles = []
        means = []
        for item in inputs:
            target = item["target"] if isinstance(item, dict) else item
            last = float(np.asarray(target)[-1])
            mean = np.full((1, prediction_length), last, dtype=np.float32)
            quantile = np.stack(
                [mean[0] + level for level in quantile_levels], axis=-1
            )[None, ...]
            means.append(mean)
            quantiles.append(quantile)
        return quantiles, means


def _adapter(**kwargs) -> tuple[ChronosAdapter, FakeChronosPipeline]:
    adapter = ChronosAdapter(device="cpu", torch_dtype="float32", **kwargs)
    pipeline = FakeChronosPipeline()
    adapter._pipeline = pipeline
    return adapter, pipeline


def test_wide_multi_series_returns_every_series_in_input_order():
    adapter, _ = _adapter()
    frame = pd.DataFrame({"B": [1.0, 2.0], "A": [8.0, 9.0]})
    result = adapter.forecast(3, frame)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], [2.0, 9.0])


@pytest.mark.parametrize("enabled", [False, True])
def test_cross_learning_is_forwarded_exactly_as_configured(enabled):
    adapter, pipeline = _adapter(cross_learning=enabled)
    adapter.forecast(2, pd.DataFrame({"A": [1.0], "B": [2.0]}))
    assert pipeline.calls[-1]["cross_learning"] is enabled
    assert len(pipeline.calls[-1]["inputs"]) == 2


def test_single_series_cross_learning_true_is_valid_and_unchanged():
    frame = pd.DataFrame({"A": [1.0, 2.0]})
    off, _ = _adapter(cross_learning=False)
    on, _ = _adapter(cross_learning=True)
    np.testing.assert_array_equal(off.forecast(3, frame), on.forecast(3, frame))


def test_context_length_is_forwarded_without_cutting_input_data():
    adapter, pipeline = _adapter(context_length=4)
    frame = pd.DataFrame({"A": np.arange(10, dtype=float)})
    adapter.forecast(2, frame)
    call = pipeline.calls[-1]
    assert call["context_length"] == 4
    assert len(call["inputs"][0]) == 10


def test_grouped_variable_histories_restore_identity_and_schema():
    adapter, _ = _adapter(cross_learning=True)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "item_id": ["A", "A", "B", "B", "B", "B"],
            "target": [3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
        }
    )
    result = adapter.forecast(
        2,
        frame,
        pipeline_type="static_grouped",
        group_col="item_id",
        time_col="date",
        target_cols=["target"],
    )
    assert list(result.columns) == ["date", "item_id", "target"]
    assert result.groupby("item_id").size().to_dict() == {"A": 2, "B": 2}
    assert result.groupby("item_id")["target"].first().to_dict() == {
        "A": 4.0,
        "B": 13.0,
    }


def test_grouped_covariates_are_sliced_per_item_and_keep_categorical_values():
    adapter, pipeline = _adapter(use_covariates=True)
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    target = pd.DataFrame(
        {
            "date": list(dates[:2]) * 2,
            "item_id": ["A", "A", "B", "B"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    future = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "item_id": ["A"] * 4 + ["B"] * 4,
            "promo": [10, 11, 12, 13, 20, 21, 22, 23],
            "weather": ["sun", "rain", "sun", "sun"] * 2,
        }
    )
    adapter.forecast(
        2,
        target,
        covariates={"future_covariates": future},
        pipeline_type="static_grouped",
        group_col="item_id",
        time_col="date",
        target_cols=["target"],
    )
    inputs = pipeline.calls[-1]["inputs"]
    np.testing.assert_array_equal(inputs[0]["future_covariates"]["promo"], [12, 13])
    np.testing.assert_array_equal(inputs[1]["future_covariates"]["promo"], [22, 23])
    assert inputs[0]["future_covariates"]["weather"].dtype == object


def test_probabilistic_multi_series_retains_item_axis():
    adapter, _ = _adapter()
    samples = adapter.forecast_probabilistic(
        4, pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}), num_samples=7
    )
    assert samples.shape == (7, 4, 2)


def test_rolling_window_batch_forces_cross_learning_off():
    adapter, pipeline = _adapter(cross_learning=True)
    result = adapter._infer_contexts([np.arange(3), np.arange(4)], horizon=2)
    assert result.shape == (2, 2)
    assert pipeline.calls[-1]["cross_learning"] is False


def test_cross_learning_fails_before_a_joint_group_is_batch_split():
    adapter, _ = _adapter(cross_learning=True, batch_size=1)
    with pytest.raises(ValueError, match="requires at least 2"):
        adapter.forecast(2, pd.DataFrame({"A": [1.0], "B": [2.0]}))
