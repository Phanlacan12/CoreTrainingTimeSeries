"""Run a configured local cold-start experiment and print comparable scores.

Examples:
    ./tsfb_env/bin/python3.11 examples/coldstart_example.py
    ./tsfb_env/bin/python3.11 examples/coldstart_example.py --offline
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "tsfb_configs" / "coldstart_default.yaml"
RESULTS_DIR = ROOT / "experiments" / "coldstart" / "results"
COLD_SERIES = "A"
METRICS = ("mae", "mse", "wape")


def run(config_path: Path, *, offline: bool) -> None:
    if offline:
        # Hugging Face reads this flag during import, so set it first.
        os.environ["HF_HUB_OFFLINE"] = "1"

    from prepare_coldstart_data import (
        DEFAULT_DONOR_HISTORY,
        DEFAULT_INPUT,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SHORT_HISTORY,
        prepare,
    )
    from tsfb.base.utils.utils import load_config
    from tsfb.run_pipeline import PipelineExecutor

    config = load_config(str(config_path))
    horizon = int(config["evaluation"]["strategy_args"]["horizon"])
    data_path = _resolve_data_path(config)
    if not data_path.exists():
        prepare(
            DEFAULT_INPUT,
            DEFAULT_OUTPUT_DIR,
            short_history=DEFAULT_SHORT_HISTORY,
            donor_history=DEFAULT_DONOR_HISTORY,
            horizon=horizon,
        )
        print(f"Prepared cold-start data under {DEFAULT_OUTPUT_DIR}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Configured data file was not generated: {data_path}. "
            "Run examples/prepare_coldstart_data.py or fix file_path in the YAML."
        )

    results_path = RESULTS_DIR / f"{config_path.stem}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    PipelineExecutor(str(config_path), str(results_path)).run()
    frame = pd.read_csv(results_path)
    summaries = _summaries(frame, COLD_SERIES)
    _print_summary(summaries, COLD_SERIES)
    _print_deltas(summaries)


def _resolve_data_path(config: dict) -> Path:
    configured = Path(
        config["evaluation"]["strategy_args"]["data_module_config"]["file_path"]
    )
    return configured if configured.is_absolute() else ROOT / configured


def _summaries(frame: pd.DataFrame, cold_series: str) -> dict:
    name_col = "approach_name"
    series_col = "metrics_per_series"
    metric_columns = {str(column).lower(): column for column in frame.columns}
    metric_cols = [metric_columns[name] for name in METRICS if name in metric_columns]
    if name_col not in frame or series_col not in frame or not metric_cols:
        raise ValueError(
            "Result file must contain approach_name, metrics_per_series and metrics."
        )

    item_rows = frame[frame[series_col].astype(str).isin(["A", "B", "C"])]
    if item_rows.empty:
        raise ValueError(
            "No per-series A/B/C metrics found; enable return_metrics_per_series."
        )

    result = {}
    for approach, rows in item_rows.groupby(name_col):
        panel = {
            str(name).lower(): value
            for name, value in rows[metric_cols].mean().to_dict().items()
        }
        cold_rows = rows[rows[series_col].astype(str) == cold_series]
        if cold_rows.empty:
            raise ValueError(f"Cold series {cold_series!r} is absent from results.")
        result[str(approach)] = {
            "panel": panel,
            "cold": {
                str(name).lower(): value
                for name, value in cold_rows[metric_cols].mean().to_dict().items()
            },
        }
    return result


def _print_summary(summaries: dict, cold_series: str) -> None:
    print("\nCold-start evaluation")
    print("- panel: macro average over A/B/C (all-series evaluation)")
    print(f"- cold:  series {cold_series} only (cold-item evaluation)\n")
    print(f"{'approach':34} {'scope':8} {'MAE':>10} {'MSE':>10} {'WAPE':>10}")
    for approach, scopes in summaries.items():
        for scope, metrics in scopes.items():
            values = " ".join(
                f"{metrics.get(name, float('nan')):10.4f}" for name in METRICS
            )
            print(f"{approach:34} {scope:8} {values}")


def _print_deltas(summaries: dict) -> None:
    pairs = [
        ("chronos_cross_off_cov_off", "chronos_cross_on_cov_off", "cross on vs off"),
        ("chronos_cross_off_cov_off", "chronos_cross_off_cov_on", "cov on vs off"),
        ("chronos_cross_on_cov_off", "chronos_cross_on_cov_on", "cov with cross"),
        ("seasonal_naive", "chronos_cross_on_cov_on", "cross+cov vs naive"),
        ("tirex_cov_off", "tirex_cov_on", "tirex cov on/off"),
        ("seasonal_naive", "tirex_cov_off", "tirex vs naive"),
        ("seasonal_naive", "tirex_cov_on", "tirex+cov vs naive"),
    ]
    available = [
        pair for pair in pairs if pair[0] in summaries and pair[1] in summaries
    ]
    if not available:
        return
    print("\nRelative error reduction (second arm against first; positive is better)")
    for reference, candidate, label in available:
        for scope in ("panel", "cold"):
            chunks = []
            for metric in METRICS:
                old = summaries[reference][scope].get(metric)
                new = summaries[candidate][scope].get(metric)
                if old is not None and new is not None and old != 0:
                    chunks.append(f"{metric.upper()}={(old - new) / old * 100:+.1f}%")
            print(f"{label:18} {scope:5}: " + ", ".join(chunks))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use an already cached Chronos checkpoint without network requests.",
    )
    args = parser.parse_args()
    run(args.config.resolve(), offline=args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
