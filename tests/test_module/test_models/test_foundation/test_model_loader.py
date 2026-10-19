"""Tests for model_loader integration with foundation.* prefix."""
from __future__ import annotations

import pytest

from tsfb.base.models.foundation.adapters.chronos_adapter import ChronosAdapter
from tsfb.base.models.model_loader import ModelFactory, get_model_info, get_models


class TestGetModelInfoFoundation:
    def test_returns_callable_for_foundation_prefix(self):
        info = get_model_info(
            {
                "model_name": "foundation.chronos",
                "model_args": {"pretrained_model_id": "autogluon/chronos-2-small"},
            }
        )
        assert callable(info)

    def test_factory_model_name_attribute(self):
        info = get_model_info({"model_name": "foundation.chronos"})
        assert hasattr(info, "model_name")
        assert info.model_name == "chronos"

    @pytest.mark.parametrize(
        "key", ["chronos", "timesfm", "tirex", "moirai", "moment", "kronos"]
    )
    def test_all_registry_keys_load(self, key):
        info = get_model_info({"model_name": f"foundation.{key}"})
        assert callable(info)

    def test_unknown_foundation_key_raises(self):
        with pytest.raises(KeyError, match="Unknown foundation model"):
            get_model_info({"model_name": "foundation.does_not_exist"})

    def test_darts_model_still_loads_unchanged(self):
        """Backward compatibility — non-foundation models unaffected."""
        import darts.models

        info = get_model_info({"model_name": "darts.models.NaiveSeasonal"})
        assert info is not None


class TestGetModelsFoundation:
    def test_returns_model_factory_list(self):
        factories = get_models(
            {
                "models": [
                    {
                        "model_name": "foundation.chronos",
                        "model_hyper_params": {
                            "pretrained_model_id": "autogluon/chronos-2-small",
                            "device": "cpu",
                        },
                    }
                ]
            }
        )
        assert len(factories) == 1
        assert isinstance(factories[0], ModelFactory)

    def test_factory_instantiates_correct_adapter(self):
        factories = get_models(
            {
                "models": [
                    {
                        "model_name": "foundation.chronos",
                        "model_hyper_params": {
                            "pretrained_model_id": "autogluon/chronos-2-small",
                            "device": "cpu",
                        },
                    }
                ]
            }
        )
        adapter = factories[0]()
        assert isinstance(adapter, ChronosAdapter)

    def test_model_hyper_params_passed_to_adapter(self):
        factories = get_models(
            {
                "models": [
                    {
                        "model_name": "foundation.chronos",
                        "model_hyper_params": {
                            "pretrained_model_id": "autogluon/chronos-2-small",
                            "device": "cpu",
                            "num_samples": 5,
                            "torch_dtype": "float32",
                        },
                    }
                ]
            }
        )
        adapter = factories[0]()
        assert adapter._num_samples == 5
        assert adapter._device == "cpu"

    def test_default_pretrained_id_from_registry(self):
        """When model_args omits pretrained_model_id, registry default is used."""
        factories = get_models({"models": [{"model_name": "foundation.chronos"}]})
        adapter = factories[0]()
        assert "chronos" in adapter._model_id.lower()

    def test_multiple_models_in_one_config(self):
        factories = get_models(
            {
                "models": [
                    {
                        "model_name": "foundation.chronos",
                        "model_hyper_params": {
                            "pretrained_model_id": "autogluon/chronos-2-small",
                            "device": "cpu",
                        },
                    },
                    {"model_name": "darts.models.NaiveSeasonal"},
                ]
            }
        )
        assert len(factories) == 2
