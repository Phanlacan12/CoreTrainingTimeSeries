"""
Foundation Model Benchmark Example

So sánh Chronos zero-shot với các baseline trên ETTh1 dataset.

Usage:
    # Chạy benchmark đầy đủ (cần chronos-forecasting):
    python examples/foundation_benchmark_example.py

    # Chỉ chạy baseline (không cần foundation model deps):
    python examples/foundation_benchmark_example.py --baseline-only

    # Chỉ test config parsing:
    python examples/foundation_benchmark_example.py --dry-run

Cài đặt trước:
    pip install "chronos-forecasting>=2.1.0"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Dependency check


def check_chronos_available() -> bool:
    try:
        import chronos  # noqa: F401

        return True
    except ImportError:
        return False


def print_install_hint() -> None:
    print(
        "\n[WARNING] chronos-forecasting not installed.\n"
        "  Install with: pip install 'chronos-forecasting>=2.1.0'\n"
        "  Foundation model approaches will be skipped.\n"
    )


# Benchmark runner


def run_benchmark(
    config_path: str,
    results_path: str,
    baseline_only: bool = False,
) -> None:
    """
    Run the pipeline benchmark and print a summary table.

    Parameters
    ----------
    config_path:   Path to the YAML benchmark config.
    results_path:  Output CSV path for raw results.
    baseline_only: Skip foundation model approaches if True.
    """
    import pandas as pd

    from tsfb.base.utils.utils import load_config
    from tsfb.run_pipeline import PipelineExecutor

    cfg = load_config(config_path)

    if baseline_only:
        cfg["model"] = {
            k: v
            for k, v in cfg.get("model", {}).items()
            if not str(v.get("model_name", "")).startswith("foundation.")
        }
        print("[INFO] baseline_only=True — foundation model approaches removed.\n")

        if not cfg["model"]:
            print("[ERROR] No models left after filtering. Exiting.")
            sys.exit(1)

        # Write patched config to a temp file
        import tempfile

        import yaml

        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, dir=ROOT
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name

    print("=" * 60)
    print("  Foundation Model Benchmark — ETTh1")
    print("=" * 60)
    print(f"  Config       : {config_path}")
    print(f"  Results CSV  : {results_path}")
    print(f"  Models       : {list(cfg.get('model', {}).keys())}")
    print("=" * 60 + "\n")

    executor = PipelineExecutor(
        config_path=config_path,
        results_path=results_path,
    )
    executor.run()

    _print_summary(results_path)


def _print_summary(results_path: str) -> None:
    """Load results CSV and print a compact leaderboard."""
    import pandas as pd

    if not Path(results_path).exists():
        print(f"[WARNING] Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Infer approach column name
    approach_col = next(
        (c for c in df.columns if "approach" in c.lower() and "name" in c.lower()),
        None,
    )

    priority_metrics = ["mae", "mse", "rmse", "mape", "smape"]
    available_metrics = [
        c
        for c in df.columns
        if c.lower() in priority_metrics and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not available_metrics:
        available_metrics = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in (approach_col or [], "horizon", "seed")
        ][:5]

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS (mean across windows)")
    print("=" * 60)

    if approach_col:
        summary = (
            df.groupby(approach_col)[available_metrics]
            .mean()
            .round(4)
            .sort_values(available_metrics[0] if available_metrics else df.columns[0])
        )
        print(summary.to_string())
        _print_winner(summary, available_metrics)
    else:
        print(df[available_metrics].mean().round(4).to_string())

    print("\nFull results saved to:", results_path)


def _print_winner(summary, metrics: list) -> None:
    if not metrics:
        return
    primary = metrics[0]
    winner = summary[primary].idxmin()
    best_val = summary.loc[winner, primary]
    print(f"\n  Best model on {primary.upper()}: {winner} ({best_val:.4f})")


# ======== Main =========


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Foundation Model Benchmark — ETTh1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "tsfb_configs" / "foundation_zeroshot_benchmark.yaml"),
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--results",
        default=str(ROOT / "artifacts" / "foundation_benchmark_results.csv"),
        help="Output CSV path for benchmark results.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip foundation model approaches (useful when deps not installed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate config only — do not run evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    chronos_ok = check_chronos_available()
    if not chronos_ok and not args.baseline_only:
        print_install_hint()
        resp = input("Continue with baseline-only mode? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted. Install chronos-forecasting and re-run.")
            return 1
        args.baseline_only = True

    if args.dry_run:
        from tsfb.base.utils.utils import load_config

        cfg = load_config(args.config)
        print("[DRY-RUN] Config parsed successfully.")
        print(f"  Models   : {list(cfg.get('model', {}).keys())}")
        strategy = cfg.get("evaluation", {}).get("strategy_args", {})
        print(
            f"  Strategy : {strategy.get('strategy_name')} | horizon={strategy.get('horizon')}"
        )
        print(f"  Data     : {cfg.get('data_module_config', {}).get('file_path')}")
        return 0

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)

    run_benchmark(
        config_path=args.config,
        results_path=args.results,
        baseline_only=args.baseline_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
