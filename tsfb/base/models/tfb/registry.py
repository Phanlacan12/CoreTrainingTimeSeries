from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, ItemsView

from tsfb.base.data_module.preprocess_model.components.window_builder import DELTA_SCALE
from tsfb.base.schema.tfb_family import ModelFamily


@dataclass(frozen=True)
class TFBModelSpec:
    """Model metadata used by the adapter and config builder."""

    public_name: str
    module_path: str
    attr_name: str
    family: ModelFamily
    defaults: Dict[str, Any] = field(default_factory=dict)

    def load_class(self) -> type:
        """Import and return the concrete model class for this spec."""
        module = import_module(self.module_path)
        return getattr(module, self.attr_name)


MODEL_REGISTRY: Dict[str, TFBModelSpec] = {
    "Autoformer": TFBModelSpec(
        public_name="Autoformer",
        module_path="tsfb.base.origin_code.time_series_library.models.Autoformer",
        attr_name="Autoformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "factor": 5, "moving_avg": 25},
    ),
    "Crossformer": TFBModelSpec(
        public_name="Crossformer",
        module_path="tsfb.base.origin_code.time_series_library.models.Crossformer",
        attr_name="Crossformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"seg_len": 2, "win_size": 2, "factor": 5},
    ),
    "Transformer": TFBModelSpec(
        public_name="Transformer",
        module_path="tsfb.base.origin_code.time_series_library.models.Transformer",
        attr_name="Transformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "factor": 5},
    ),
    "Informer": TFBModelSpec(
        public_name="Informer",
        module_path="tsfb.base.origin_code.time_series_library.models.Informer",
        attr_name="Informer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "factor": 5, "distil": True},
    ),
    "Reformer": TFBModelSpec(
        public_name="Reformer",
        module_path="tsfb.base.origin_code.time_series_library.models.Reformer",
        attr_name="Reformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "factor": 5},
    ),
    "TiDE": TFBModelSpec(
        public_name="TiDE",
        module_path="tsfb.base.origin_code.time_series_library.models.TiDE",
        attr_name="TiDE",
        family=ModelFamily.SEQ2SEQ,
        defaults={"freq": "d"},
    ),
    "DLinear": TFBModelSpec(
        public_name="DLinear",
        module_path="tsfb.base.origin_code.time_series_library.models.DLinear",
        attr_name="DLinear",
        family=ModelFamily.SEQ2SEQ,
        defaults={"moving_avg": 25},
    ),
    "Nonstationary_Transformer": TFBModelSpec(
        public_name="Nonstationary_Transformer",
        module_path=(
            "tsfb.base.origin_code.time_series_library.models.Nonstationary_Transformer"
        ),
        attr_name="Nonstationary_Transformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={
            "embed": "timeF",
            "factor": 5,
            "p_hidden_dims": [64, 64],
            "p_hidden_layers": 2,
        },
    ),
    "FEDformer": TFBModelSpec(
        public_name="FEDformer",
        module_path="tsfb.base.origin_code.time_series_library.models.FEDformer",
        attr_name="FEDformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "moving_avg": 25},
    ),
    "ETSformer": TFBModelSpec(
        public_name="ETSformer",
        module_path="tsfb.base.origin_code.time_series_library.models.ETSformer",
        attr_name="ETSformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "top_k": 5},
    ),
    "Pyraformer": TFBModelSpec(
        public_name="Pyraformer",
        module_path="tsfb.base.origin_code.time_series_library.models.Pyraformer",
        attr_name="Pyraformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={},
    ),
    "PatchTST": TFBModelSpec(
        public_name="PatchTST",
        module_path="tsfb.base.origin_code.time_series_library.models.PatchTST",
        attr_name="PatchTST",
        family=ModelFamily.SEQ2SEQ,
        defaults={"patch_len": 16, "stride": 8, "factor": 5},
    ),
    "MICN": TFBModelSpec(
        public_name="MICN",
        module_path="tsfb.base.origin_code.time_series_library.models.MICN",
        attr_name="MICN",
        family=ModelFamily.SEQ2SEQ,
        defaults={"conv_kernel": [2, 4], "embed": "timeF"},
    ),
    "TimesNet": TFBModelSpec(
        public_name="TimesNet",
        module_path="tsfb.base.origin_code.time_series_library.models.TimesNet",
        attr_name="TimesNet",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "top_k": 5, "num_kernels": 6},
    ),
    "FiLM": TFBModelSpec(
        public_name="FiLM",
        module_path="tsfb.base.origin_code.time_series_library.models.FiLM",
        attr_name="FiLM",
        family=ModelFamily.SEQ2SEQ,
        defaults={"modes": 32, "ratio": 0.5, "output_attention": False},
    ),
    "Koopa": TFBModelSpec(
        public_name="Koopa",
        module_path="tsfb.base.origin_code.time_series_library.models.Koopa",
        attr_name="Koopa",
        family=ModelFamily.SEQ2SEQ,
        defaults={},
    ),
    "TSMixer": TFBModelSpec(
        public_name="TSMixer",
        module_path="tsfb.base.origin_code.time_series_library.models.TSMixer",
        attr_name="TSMixer",
        family=ModelFamily.SEQ2SEQ,
        defaults={},
    ),
    "SegRNN": TFBModelSpec(
        public_name="SegRNN",
        module_path="tsfb.base.origin_code.time_series_library.models.SegRNN",
        attr_name="SegRNN",
        family=ModelFamily.SEQ2SEQ,
        defaults={"seg_len": 2},
    ),
    "FreTS": TFBModelSpec(
        public_name="FreTS",
        module_path="tsfb.base.origin_code.time_series_library.models.FreTS",
        attr_name="FreTS",
        family=ModelFamily.SEQ2SEQ,
        defaults={"channel_independence": "0"},
    ),
    "LightTS": TFBModelSpec(
        public_name="LightTS",
        module_path="tsfb.base.origin_code.time_series_library.models.LightTS",
        attr_name="LightTS",
        family=ModelFamily.SEQ2SEQ,
        defaults={},
    ),
    "TemporalFusionTransformer": TFBModelSpec(
        public_name="TemporalFusionTransformer",
        module_path=(
            "tsfb.base.origin_code.time_series_library.models.TemporalFusionTransformer"
        ),
        attr_name="TemporalFusionTransformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"data": "ETTh1", "embed": "timeF"},
    ),
    "TimeMixer": TFBModelSpec(
        public_name="TimeMixer",
        module_path="tsfb.base.origin_code.time_series_library.models.TimeMixer",
        attr_name="TimeMixer",
        family=ModelFamily.SEQ2SEQ,
        defaults={
            "moving_avg": 25,
            "embed": "timeF",
            "down_sampling_window": 2,
            "down_sampling_layers": 1,
            "channel_independence": 0,
            "decomp_method": "moving_avg",
            "top_k": 5,
        },
    ),
    "iTransformer": TFBModelSpec(
        public_name="iTransformer",
        module_path="tsfb.base.origin_code.time_series_library.models.iTransformer",
        attr_name="iTransformer",
        family=ModelFamily.SEQ2SEQ,
        defaults={"embed": "timeF", "factor": 5},
    ),
    "TimeKAN": TFBModelSpec(
        public_name="TimeKAN",
        module_path="tsfb.base.origin_code.timekan.models.timekan_model",
        attr_name="TimeKANModeL",
        family=ModelFamily.SINGLE_INPUT,
        defaults={
            "moving_avg": 25,
            "embed": "timeF",
            "down_sampling_window": 2,
            "down_sampling_layers": 1,
            "use_future_temporal_feature": False,
            "begin_order": 2,
        },
    ),
    "xPatch": TFBModelSpec(
        public_name="xPatch",
        module_path="tsfb.base.origin_code.xpatch.models.xPatch",
        attr_name="xPatchModel",
        family=ModelFamily.SINGLE_INPUT,
        defaults={
            "patch_len": 16,
            "stride": 8,
            "padding_patch": "end",
            "revin": True,
            "ma_type": "reg",
            "alpha": 0.5,
            "beta": 0.5,
        },
    ),
    "PDF": TFBModelSpec(
        public_name="PDF",
        module_path="tsfb.base.origin_code.pdf.models.PDF",
        attr_name="Model",
        family=ModelFamily.SINGLE_INPUT,
        defaults={
            "patch_len": 16,
            "stride": 8,
            "kernel_list": [3, 5],
            "period": 24,
            "fc_dropout": 0.1,
            "head_dropout": 0.1,
            "individual": False,
            "add": False,
            "wo_conv": False,
            "serial_conv": False,
            "padding_patch": "end",
            "revin": True,
            "affine": True,
            "subtract_last": False,
        },
    ),
    "FITS": TFBModelSpec(
        public_name="FITS",
        module_path="tsfb.base.origin_code.fits.fits_model",
        attr_name="FITSModel",
        family=ModelFamily.SINGLE_INPUT,
        defaults={"individual": False, "cut_freq": 8},
    ),
    "DUET": TFBModelSpec(
        public_name="DUET",
        module_path="tsfb.base.origin_code.duet.models.duet_model",
        attr_name="DUETModel",
        family=ModelFamily.SINGLE_INPUT,
        defaults={
            "CI": False,
            "factor": 5,
            "fc_dropout": 0.1,
            "output_attention": False,
        },
    ),
    "Amplifier": TFBModelSpec(
        public_name="Amplifier",
        module_path="tsfb.base.origin_code.amplifier.models.amplifier_model",
        attr_name="AmplifierModel",
        family=ModelFamily.SEQ2SEQ,
        defaults={"hidden_size": 64},
    ),
    "PatchMLP": TFBModelSpec(
        public_name="PatchMLP",
        module_path="tsfb.base.origin_code.patchmlp.models.patchmlp_model",
        attr_name="PatchMLPModel",
        family=ModelFamily.SEQ2SEQ,
        defaults={"output_attention": False},
    ),
    "HDMixer": TFBModelSpec(
        public_name="HDMixer",
        module_path="tsfb.base.origin_code.hdmixer.models.HDMixer",
        attr_name="HDMixerModel",
        family=ModelFamily.SINGLE_INPUT,
        defaults={
            "patch_len": 16,
            "stride": 8,
            "padding_patch": "end",
            "revin": True,
            "fc_dropout": 0.1,
            "head_dropout": 0.1,
            "individual": False,
            "decomposition": False,
            "kernel_size": 25,
            "deform_patch": False,
            "deform_range": 0.5,
            "lambda_": 0.1,
            "r": 0.5,
            "mix_time": True,
            "mix_variable": True,
            "mix_channel": True,
        },
    ),
    "GRUD": TFBModelSpec(
        public_name="GRUD",
        module_path="tsfb.base.origin_code.ismts.forecasters",
        attr_name="GRUDForecaster",
        family=ModelFamily.MASKED_DECAY,
        defaults={"delta_scale": DELTA_SCALE},
    ),
    "TLSTM": TFBModelSpec(
        public_name="TLSTM",
        module_path="tsfb.base.origin_code.ismts.forecasters",
        attr_name="TLSTMForecaster",
        family=ModelFamily.MASKED_DECAY,
        defaults={"delta_scale": DELTA_SCALE},
    ),
    "BRITS": TFBModelSpec(
        public_name="BRITS",
        module_path="tsfb.base.origin_code.ismts.forecasters",
        attr_name="BRITSForecaster",
        family=ModelFamily.MASKED_DECAY,
        defaults={"delta_scale": DELTA_SCALE, "consistency_weight": 0.1},
    ),
}


def get_model_spec(model_key: str) -> TFBModelSpec:
    """Return the registered model spec for a benchmark-facing model key."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown TFB model {model_key!r}. Supported: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_key]


def iter_model_specs() -> ItemsView[str, TFBModelSpec]:
    """Iterate over all registered TFB model specs."""
    return MODEL_REGISTRY.items()
