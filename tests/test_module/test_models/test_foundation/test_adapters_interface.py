"""
Interface contract tests for all foundation model adapters.

Verifies that each adapter:
  - Is instantiable without loading weights (lazy)
  - Exposes correct ModelBase interface
  - Has correct capability flags matching the registry
  - Stores config params correctly

Heavy tests (actual model inference) are in test_chronos_adapter.py.
Other adapters skip if their deps are not installed.
"""
from __future__ import annotations

import pytest

from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.models.foundation.registry import FOUNDATION_MODEL_REGISTRY


def _get_adapter_class(key: str):
    """Import the adapter class for a registry key."""
    import importlib

    spec = FOUNDATION_MODEL_REGISTRY[key]
    module_path, class_name = spec.adapter_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ── Chronos (always available) ────────────────────────────────────────────────


class TestChronosAdapterInterface:
    def test_is_foundation_model_base(self):
        from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

        assert issubclass(ChronosAdapter, FoundationModelBase)

    def test_instantiates_without_loading_weights(self):
        from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

        a = ChronosAdapter(pretrained_model_id="autogluon/chronos-2-small")
        assert a._pipeline is None

    def test_model_name_contains_model_id(self):
        from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

        a = ChronosAdapter(pretrained_model_id="autogluon/chronos-2-small")
        assert "autogluon/chronos-2-small" in a.model_name

    def test_capability_matches_registry(self):
        from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

        spec = FOUNDATION_MODEL_REGISTRY["chronos"]
        a = ChronosAdapter(pretrained_model_id="x")
        assert a.supports_finetuning == spec.supports_finetuning
        assert a.supports_probabilistic == spec.supports_probabilistic

    def test_accepts_finetune_config_as_dict(self):
        from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

        a = ChronosAdapter(
            pretrained_model_id="x",
            finetune_config={"mode": "lora", "epochs": 5},
        )
        assert a._raw_finetune_config["epochs"] == 5

    def test_kwargs_ignored_gracefully(self):
        from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

        ChronosAdapter(pretrained_model_id="x", unknown_param="ignored")


# ── TimesFM ───────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    pytest.importorskip("timesfm", reason="timesfm not installed") is None,
    reason="timesfm not installed",
)
class TestTimesFMAdapterInterface:
    def test_is_foundation_model_base(self):
        from tsfb.base.models.foundation.adapters.timesfm_adapter import TimesFMAdapter

        assert issubclass(TimesFMAdapter, FoundationModelBase)

    def test_instantiates_lazy(self):
        from tsfb.base.models.foundation.adapters.timesfm_adapter import TimesFMAdapter

        a = TimesFMAdapter(pretrained_model_id="google/timesfm-1.0-200m-pytorch")
        assert a._model is None

    def test_model_name(self):
        from tsfb.base.models.foundation.adapters.timesfm_adapter import TimesFMAdapter

        a = TimesFMAdapter(pretrained_model_id="google/timesfm-1.0-200m-pytorch")
        assert "timesfm" in a.model_name.lower()


# ── Moirai ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    pytest.importorskip("uni2ts", reason="uni2ts not installed") is None,
    reason="uni2ts not installed",
)
class TestMoiraiAdapterInterface:
    def test_is_foundation_model_base(self):
        from tsfb.base.models.foundation.adapters.moirai_adapter import MoiraiAdapter

        assert issubclass(MoiraiAdapter, FoundationModelBase)

    def test_instantiates_lazy(self):
        from tsfb.base.models.foundation.adapters.moirai_adapter import MoiraiAdapter

        a = MoiraiAdapter(pretrained_model_id="Salesforce/moirai-1.0-R-small")
        assert a._pipeline is None

    def test_supports_probabilistic(self):
        from tsfb.base.models.foundation.adapters.moirai_adapter import MoiraiAdapter

        a = MoiraiAdapter(pretrained_model_id="x")
        assert a.supports_probabilistic is True


# ── MOMENT ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    pytest.importorskip("momentfm", reason="momentfm not installed") is None,
    reason="momentfm not installed",
)
class TestMomentAdapterInterface:
    def test_is_foundation_model_base(self):
        from tsfb.base.models.foundation.adapters.moment_adapter import MomentAdapter

        assert issubclass(MomentAdapter, FoundationModelBase)

    def test_instantiates_lazy(self):
        from tsfb.base.models.foundation.adapters.moment_adapter import MomentAdapter

        a = MomentAdapter(pretrained_model_id="AutonLab/MOMENT-1-large")
        assert a._model is None


# ── Kronos ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    pytest.importorskip("kronos", reason="kronos not installed") is None,
    reason="kronos not installed",
)
class TestKronosAdapterInterface:
    def test_is_foundation_model_base(self):
        from tsfb.base.models.foundation.adapters.kronos_adapter import KronosAdapter

        assert issubclass(KronosAdapter, FoundationModelBase)

    def test_instantiates_lazy(self):
        from tsfb.base.models.foundation.adapters.kronos_adapter import KronosAdapter

        a = KronosAdapter(pretrained_model_id="shiyu-coder/Kronos-Base")
        assert a._model is None


# ── Generic adapter attribute contract (Chronos only for now) ─────────────────


@pytest.mark.parametrize(
    "attr",
    ["_model_id", "_device", "_inference_mode", "_raw_finetune_config"],
)
def test_chronos_adapter_has_expected_attributes(attr):
    from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter

    adapter = ChronosAdapter(pretrained_model_id="any")
    assert hasattr(adapter, attr), f"Missing attribute: {attr}"
