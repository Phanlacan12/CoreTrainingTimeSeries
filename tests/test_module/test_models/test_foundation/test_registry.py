"""Tests for FOUNDATION_MODEL_REGISTRY and get_foundation_spec."""
from __future__ import annotations

import pytest

from tsfb.base.models.foundation.registry import (
    FOUNDATION_MODEL_REGISTRY,
    FoundationModelSpec,
    get_foundation_spec,
)

EXPECTED_KEYS = {"chronos", "timesfm", "tirex", "moirai", "moment", "kronos"}


class TestFoundationModelRegistry:
    def test_all_expected_keys_present(self):
        assert EXPECTED_KEYS.issubset(FOUNDATION_MODEL_REGISTRY.keys())

    def test_spec_type(self):
        for spec in FOUNDATION_MODEL_REGISTRY.values():
            assert isinstance(spec, FoundationModelSpec)

    def test_adapter_class_format(self):
        for key, spec in FOUNDATION_MODEL_REGISTRY.items():
            parts = spec.adapter_class.split(".")
            assert len(parts) >= 2, f"{key}: adapter_class must be a dotted path"
            assert parts[-1][0].isupper(), f"{key}: class name must start uppercase"

    def test_default_model_id_not_empty(self):
        for key, spec in FOUNDATION_MODEL_REGISTRY.items():
            assert spec.default_model_id, f"{key}: default_model_id must not be empty"

    @pytest.mark.parametrize("key", list(EXPECTED_KEYS))
    def test_get_foundation_spec_returns_correct_type(self, key):
        spec = get_foundation_spec(key)
        assert isinstance(spec, FoundationModelSpec)

    @pytest.mark.parametrize("key", list(EXPECTED_KEYS))
    def test_get_foundation_spec_case_insensitive(self, key):
        spec_lower = get_foundation_spec(key.lower())
        spec_upper = get_foundation_spec(key.upper())
        assert spec_lower.adapter_class == spec_upper.adapter_class

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown foundation model"):
            get_foundation_spec("nonexistent_model")

    def test_chronos_supports_probabilistic(self):
        spec = get_foundation_spec("chronos")
        assert spec.supports_probabilistic is True

    def test_chronos_supports_finetuning(self):
        spec = get_foundation_spec("chronos")
        assert spec.supports_finetuning is True

    def test_install_hints_present(self):
        for key, spec in FOUNDATION_MODEL_REGISTRY.items():
            assert spec.install_hint, f"{key}: install_hint should be provided"
