from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from tsfb.base.models.tfb.registry import TFBModelSpec

PASSTHROUGH_KEYS = (
    "moving_avg",
    "patch_len",
    "stride",
    "padding_patch",
    "revin",
    "ma_type",
    "alpha",
    "beta",
    "individual",
    "fc_dropout",
    "head_dropout",
    "kernel_list",
    "period",
    "add",
    "wo_conv",
    "serial_conv",
    "affine",
    "subtract_last",
    "cut_freq",
    "CI",
    "hidden_size",
    "down_sampling_window",
    "down_sampling_layers",
    "channel_independence",
    "decomp_method",
    "top_k",
    "begin_order",
    "use_future_temporal_feature",
    "use_norm",
    "noisy_gating",
    "num_experts",
    "k",
    "SCI",
    "down_sampling_method",
    "decomposition",
    "kernel_size",
    "deform_patch",
    "deform_range",
    "lambda_",
    "r",
    "mix_time",
    "mix_variable",
    "mix_channel",
    "seg_len",
    "win_size",
    "distil",
    "conv_kernel",
    "p_hidden_dims",
    "p_hidden_layers",
    "num_kernels",
    "modes",
    "ratio",
    "data",
    "device",
)


def _normalized_freq(freq: Any) -> str:
    if not freq:
        return "D"
    freq_str = str(freq).lower()
    mapping = {
        "h": "h",
        "hour": "h",
        "hourly": "h",
        "d": "d",
        "day": "d",
        "daily": "d",
        "w": "w",
        "weekly": "w",
        "m": "m",
        "month": "m",
        "monthly": "m",
        "min": "t",
        "t": "t",
    }
    return mapping.get(freq_str, freq_str[:1])


def _require_chunk_lengths(
    spec: TFBModelSpec,
    model_hyper_params: Dict[str, Any],
) -> tuple[int, int, int]:
    if "input_chunk_length" not in model_hyper_params:
        raise ValueError(
            f"Model {spec.public_name} requires"
            " 'input_chunk_length' in model_hyper_params."
        )
    if "output_chunk_length" not in model_hyper_params:
        raise ValueError(
            f"Model {spec.public_name} requires "
            f"'output_chunk_length' in model_hyper_params."
        )

    seq_len = int(model_hyper_params["input_chunk_length"])
    pred_len = int(model_hyper_params["output_chunk_length"])
    label_len = int(model_hyper_params.get("label_len", max(1, seq_len // 2)))
    return seq_len, pred_len, label_len


# pylint: disable-next=too-many-arguments
def _build_base_config(
    spec: TFBModelSpec,
    model_hyper_params: Dict[str, Any],
    *,
    num_target_channels: int,
    data_freq: Any,
    seq_len: int,
    pred_len: int,
    label_len: int,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(spec.defaults)
    cfg.update(
        {
            "task_name": "long_term_forecast",
            "seq_len": seq_len,
            "pred_len": pred_len,
            "horizon": pred_len,
            "label_len": label_len,
            "enc_in": num_target_channels,
            "dec_in": num_target_channels,
            "c_out": num_target_channels,
            "embed": model_hyper_params.get("embed", "timeF"),
            "freq": _normalized_freq(model_hyper_params.get("freq", data_freq)),
            "dropout": float(model_hyper_params.get("dropout", 0.1)),
            "d_model": int(model_hyper_params.get("d_model", 64)),
            "d_ff": int(model_hyper_params.get("d_ff", 128)),
            "e_layers": int(
                model_hyper_params.get(
                    "e_layers",
                    model_hyper_params.get("num_encoder_layers", 2),
                )
            ),
            "d_layers": int(
                model_hyper_params.get(
                    "d_layers",
                    model_hyper_params.get("num_decoder_layers", 1),
                )
            ),
            "n_heads": int(
                model_hyper_params.get(
                    "n_heads",
                    model_hyper_params.get("nhead", 4),
                )
            ),
            "factor": int(model_hyper_params.get("factor", 5)),
            "activation": model_hyper_params.get("activation", "gelu"),
            "output_attention": bool(model_hyper_params.get("output_attention", False)),
            "moving_avg": int(model_hyper_params.get("moving_avg", 25)),
            "use_norm": int(model_hyper_params.get("use_norm", 1)),
            "hidden_size": int(
                model_hyper_params.get(
                    "hidden_size",
                    model_hyper_params.get("d_model", 64),
                )
            ),
            "channel_independence": int(
                model_hyper_params.get("channel_independence", 0)
            ),
            "down_sampling_method": model_hyper_params.get(
                "down_sampling_method",
                "avg",
            ),
            "noisy_gating": bool(model_hyper_params.get("noisy_gating", False)),
            "num_experts": int(model_hyper_params.get("num_experts", 4)),
            "k": int(model_hyper_params.get("k", 2)),
            "SCI": bool(model_hyper_params.get("SCI", False)),
            "decomposition": bool(model_hyper_params.get("decomposition", False)),
            "kernel_size": int(model_hyper_params.get("kernel_size", 25)),
            "deform_patch": bool(model_hyper_params.get("deform_patch", False)),
            "deform_range": float(model_hyper_params.get("deform_range", 0.5)),
            "lambda_": float(model_hyper_params.get("lambda_", 0.1)),
            "r": float(model_hyper_params.get("r", 0.5)),
            "mix_time": bool(model_hyper_params.get("mix_time", True)),
            "mix_variable": bool(model_hyper_params.get("mix_variable", True)),
            "mix_channel": bool(model_hyper_params.get("mix_channel", True)),
            "revin": bool(model_hyper_params.get("revin", True)),
            "affine": bool(model_hyper_params.get("affine", True)),
            "subtract_last": bool(model_hyper_params.get("subtract_last", False)),
            "seg_len": int(model_hyper_params.get("seg_len", max(1, pred_len))),
            "win_size": int(model_hyper_params.get("win_size", 2)),
            "distil": bool(model_hyper_params.get("distil", True)),
            "conv_kernel": model_hyper_params.get("conv_kernel", [2, 4]),
            "p_hidden_dims": model_hyper_params.get("p_hidden_dims", [64, 64]),
            "p_hidden_layers": int(model_hyper_params.get("p_hidden_layers", 2)),
            "num_kernels": int(model_hyper_params.get("num_kernels", 6)),
            "modes": int(model_hyper_params.get("modes", 32)),
            "ratio": float(model_hyper_params.get("ratio", 0.5)),
            "data": model_hyper_params.get("data", "ETTh1"),
            "device": model_hyper_params.get("device", "cpu"),
        }
    )
    return cfg


def _apply_passthrough_keys(
    cfg: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
) -> None:
    for key in PASSTHROUGH_KEYS:
        if key in model_hyper_params:
            cfg[key] = model_hyper_params[key]


def _configure_pdf(cfg: Dict[str, Any], model_hyper_params: Dict[str, Any]) -> None:
    periods = model_hyper_params.get("period", cfg.get("period", [24]))
    if not isinstance(periods, (list, tuple)):
        periods = [int(periods)]
    cfg["period"] = [int(period) for period in periods]

    patch_len = model_hyper_params.get("patch_len", cfg.get("patch_len", 16))
    if not isinstance(patch_len, (list, tuple)):
        patch_len = [int(patch_len)] * len(cfg["period"])
    cfg["patch_len"] = [int(value) for value in patch_len]

    stride = model_hyper_params.get("stride", cfg.get("stride", 8))
    if stride is None:
        cfg["stride"] = None
    else:
        if not isinstance(stride, (list, tuple)):
            stride = [int(stride)] * len(cfg["period"])
        cfg["stride"] = [int(value) for value in stride]

    kernel_list = model_hyper_params.get(
        "kernel_list",
        cfg.get("kernel_list", [3, 5]),
    )
    if not isinstance(kernel_list, (list, tuple)):
        kernel_list = [int(kernel_list)]
    cfg["kernel_list"] = [int(value) for value in kernel_list]


def _configure_patch_mlp(
    cfg: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
    seq_len: int,
) -> None:
    patch_len = model_hyper_params.get("patch_len")
    if patch_len is None:
        patch_len = [
            seq_len,
            max(2, seq_len // 2),
            max(2, seq_len // 4),
            max(2, seq_len // 8),
        ]
    elif not isinstance(patch_len, (list, tuple)):
        patch_len = [int(patch_len)] * 4
    cfg["patch_len"] = [max(2, min(seq_len, int(value))) for value in patch_len]


def _configure_patch_tst(
    cfg: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
    seq_len: int,
) -> None:
    patch_len = int(model_hyper_params.get("patch_len", max(2, seq_len // 2)))
    patch_len = max(2, min(seq_len, patch_len))
    cfg["patch_len"] = patch_len
    cfg["stride"] = int(model_hyper_params.get("stride", max(1, patch_len // 2)))


def _configure_etsformer(
    cfg: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
    seq_len: int,
) -> None:
    if (
        "d_layers" not in model_hyper_params
        and "num_decoder_layers" not in model_hyper_params
    ):
        cfg["d_layers"] = cfg["e_layers"]
    max_k = max(1, (seq_len // 2) - (1 if seq_len % 2 == 0 else 0))
    cfg["top_k"] = max(1, min(int(cfg["top_k"]), max_k))


def _configure_segment_models(
    cfg: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
    *,
    seq_len: int,
    pred_len: int,
) -> None:
    default_seg_len = max(1, pred_len)
    if seq_len % default_seg_len != 0:
        default_seg_len = 1
    cfg["seg_len"] = int(model_hyper_params.get("seg_len", default_seg_len))


def _apply_model_specific_overrides(
    spec: TFBModelSpec,
    cfg: Dict[str, Any],
    model_hyper_params: Dict[str, Any],
    *,
    seq_len: int,
    pred_len: int,
) -> None:
    if spec.public_name == "PDF":
        _configure_pdf(cfg, model_hyper_params)
    if spec.public_name == "PatchMLP":
        _configure_patch_mlp(cfg, model_hyper_params, seq_len)
    if spec.public_name == "PatchTST":
        _configure_patch_tst(cfg, model_hyper_params, seq_len)
    if spec.public_name == "ETSformer":
        _configure_etsformer(cfg, model_hyper_params, seq_len)
    if spec.public_name == "FreTS":
        cfg["channel_independence"] = str(
            model_hyper_params.get(
                "channel_independence",
                cfg.get("channel_independence", "0"),
            )
        )
    if spec.public_name in {"Crossformer", "SegRNN"}:
        _configure_segment_models(
            cfg,
            model_hyper_params,
            seq_len=seq_len,
            pred_len=pred_len,
        )
    if spec.public_name == "Autoformer" and "factor" not in model_hyper_params:
        cfg["factor"] = 1
    if spec.public_name == "FEDformer" and "n_heads" not in model_hyper_params:
        cfg["n_heads"] = 8


def _normalize_common_config(cfg: Dict[str, Any], seq_len: int) -> None:
    if "moving_avg" in cfg:
        moving_avg = int(cfg["moving_avg"])
        max_kernel = seq_len if seq_len % 2 == 1 else max(1, seq_len - 1)
        moving_avg = max(1, min(moving_avg, max_kernel))
        if moving_avg % 2 == 0:
            moving_avg = max(1, moving_avg - 1)
        cfg["moving_avg"] = moving_avg

    if "top_k" in cfg:
        cfg["top_k"] = max(1, min(int(cfg["top_k"]), max(1, seq_len // 2)))


def build_config_namespace(
    spec: TFBModelSpec,
    model_hyper_params: Dict[str, Any],
    *,
    num_target_channels: int,
    data_freq: Any = None,
) -> SimpleNamespace:
    """Build a repo-native configuration object expected by origin models."""
    seq_len, pred_len, label_len = _require_chunk_lengths(spec, model_hyper_params)
    cfg = _build_base_config(
        spec,
        model_hyper_params,
        num_target_channels=num_target_channels,
        data_freq=data_freq,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
    )
    _apply_passthrough_keys(cfg, model_hyper_params)
    _apply_model_specific_overrides(
        spec,
        cfg,
        model_hyper_params,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    _normalize_common_config(cfg, seq_len)

    return SimpleNamespace(**cfg)
