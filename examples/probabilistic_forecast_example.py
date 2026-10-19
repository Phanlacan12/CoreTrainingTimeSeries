"""
Probabilistic Forecast Example
================================
Minh họa forecast_probabilistic() và forecast_distribution() trên nhiều loại model:

  Statistic models:  PoissonGLM, NegativeBinomialGLM, ProphetStatistic
  Darts models:      LinearRegressionModel (deterministic),
                     NBEATSModel + GaussianLikelihood (probabilistic)
  TFB models:        DLinear (deterministic → base-class Normal fallback)
  Foundation models: ChronosAdapter (quantile-based),
                     TimesFMAdapter (quantile-based),
                     MoiraiAdapter (MC samples, nếu có uni2ts)

Chạy:
    python examples/probabilistic_forecast_example.py            # tất cả
    python examples/probabilistic_forecast_example.py --model poisson
    python examples/probabilistic_forecast_example.py --model darts
    python examples/probabilistic_forecast_example.py --model tfb
    python examples/probabilistic_forecast_example.py --model foundation
    python examples/probabilistic_forecast_example.py --model chronos --dist student_t
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON = 24
NUM_SAMPLES = 200
TRAIN_N = 720
DATA_PATH = ROOT / "tests" / "data" / "ETTh1.csv"


def load_series(col: str = "OT") -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    return df[[col]].iloc[:TRAIN_N]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def show_samples(samples: np.ndarray, label: str) -> None:
    q = np.percentile(samples, [10, 50, 90], axis=0)
    print(f"  [{label}]")
    print(f"    shape         : {samples.shape}")
    print(f"    q10/q50/q90   : {q[0,0]:.2f} / {q[1,0]:.2f} / {q[2,0]:.2f}  (step 1)")
    print(f"    mean interval : {np.mean(q[2]-q[0]):.2f}")


def show_dist(df_fc, label: str) -> None:
    print(f"  [{label}]  dist={df_fc.distribution_name}")
    print(f"    params   : {list(df_fc.params)}")
    print(f"    mean[0]  : {df_fc.mean[0]:.2f}")
    print(f"    std[0]   : {float(np.sqrt(df_fc.variance[0])):.2f}")
    q10, q90 = df_fc.quantile(0.1), df_fc.quantile(0.9)
    print(f"    q10/q90  : {q10[0]:.2f} / {q90[0]:.2f}")
    print(f"    sample() : {df_fc.sample(50).shape}")


# ---------------------------------------------------------------------------
# Statistic models
# ---------------------------------------------------------------------------


def run_poisson(series: pd.DataFrame) -> None:
    section("PoissonGLM  — exact Poisson(mu) distribution")
    from tsfb.base.models.statistic_impl.poisson_glm import PoissonGLM

    m = PoissonGLM(alpha=0.05)
    m.forecast_fit(series.abs())

    print("\n  forecast_probabilistic():")
    show_samples(
        m.forecast_probabilistic(HORIZON, series.abs(), NUM_SAMPLES), "Poisson samples"
    )
    print("\n  forecast_distribution():")
    show_dist(m.forecast_distribution(HORIZON, series.abs()), "Poisson(mu)")


def run_negbinom(series: pd.DataFrame) -> None:
    section("NegativeBinomialGLM  — exact NegBinomial(mu, alpha) distribution")
    from tsfb.base.models.statistic_impl.negative_binomial import NegativeBinomialGLM

    m = NegativeBinomialGLM(alpha=0.05, r=5.0)
    m.forecast_fit(series.abs())

    print("\n  forecast_probabilistic():")
    show_samples(
        m.forecast_probabilistic(HORIZON, series.abs(), NUM_SAMPLES),
        "NegBinomial samples",
    )
    print("\n  forecast_distribution():")
    show_dist(m.forecast_distribution(HORIZON, series.abs()), "NegBinomial(mu, alpha)")


def run_prophet(series: pd.DataFrame) -> None:
    section("ProphetStatistic  — Normal fitted to Prophet uncertainty intervals")
    try:
        from tsfb.base.models.statistic_impl.prophet_statistic import ProphetStatistic
    except ImportError:
        print("  [SKIP] pip install prophet")
        return

    m = ProphetStatistic(alpha=0.05)
    m.forecast_fit(series)

    print("\n  forecast_probabilistic():")
    show_samples(
        m.forecast_probabilistic(HORIZON, series, NUM_SAMPLES), "Prophet Normal samples"
    )
    print("\n  forecast_distribution():")
    show_dist(m.forecast_distribution(HORIZON, series), "Normal(yhat, sigma)")


# ---------------------------------------------------------------------------
# Darts models
# ---------------------------------------------------------------------------


def run_darts(series: pd.DataFrame) -> None:
    section("Darts models")
    try:
        from darts.models import (
            ExponentialSmoothing,
            LinearRegressionModel,
            NBEATSModel,
        )
        from darts.utils.likelihood_models import GaussianLikelihood
    except ImportError:
        print("  [SKIP] pip install darts")
        return

    from tsfb.base.models.darts.darts_models import DartsModelAdapter

    # --- ExponentialSmoothing: no likelihood --------------------------------
    print("\n  [ExponentialSmoothing]  deterministic, no likelihood")
    es = DartsModelAdapter(model_class=ExponentialSmoothing, model_args={})
    es.forecast_fit(train_data=series)
    print(f"    supports_probabilistic : {es.supports_probabilistic}")
    print(f"    forecast_probabilistic : {es.forecast_probabilistic(HORIZON, series)}")

    # --- LinearRegression: no likelihood ------------------------------------
    print("\n  [LinearRegressionModel]  deterministic, no likelihood")
    lr = DartsModelAdapter(
        model_class=LinearRegressionModel,
        model_args={"lags": 48, "output_chunk_length": HORIZON},
    )
    lr.forecast_fit(train_data=series)
    print(f"    supports_probabilistic : {lr.supports_probabilistic}")
    print(f"    forecast_probabilistic : {lr.forecast_probabilistic(HORIZON, series)}")

    # --- NBEATSModel + GaussianLikelihood: probabilistic -------------------
    print("\n  [NBEATSModel + GaussianLikelihood]  probabilistic (2 epochs)")
    nbeats = DartsModelAdapter(
        model_class=NBEATSModel,
        model_args={
            "input_chunk_length": 48,
            "output_chunk_length": HORIZON,
            "n_epochs": 2,
            "batch_size": 64,
            "random_state": 42,
            "likelihood": GaussianLikelihood(),
            "pl_trainer_kwargs": {"enable_progress_bar": False},
        },
    )
    nbeats.forecast_fit(train_data=series)
    print(f"    supports_probabilistic : {nbeats.supports_probabilistic}")

    print("\n    forecast_probabilistic():")
    samples = nbeats.forecast_probabilistic(HORIZON, series, num_samples=NUM_SAMPLES)
    if samples is not None:
        show_samples(samples, "NBEATS+Gaussian samples")

    print("\n    forecast_distribution() — Normal moment-matched from samples:")
    dist_fc = nbeats.forecast_distribution(HORIZON, series, num_samples=NUM_SAMPLES)
    if dist_fc is not None:
        show_dist(dist_fc, "NBEATS → Normal")

    print("\n    to_quantile_dfs() — format for strategy layer:")
    if dist_fc is not None:
        q_dfs = dist_fc.to_quantile_dfs(
            [0.1, 0.5, 0.9],
            index=pd.RangeIndex(HORIZON),
            target_cols=["OT"],
        )
        for k, df in q_dfs.items():
            print(f"      {k}: step1={df.iloc[0, 0]:.2f}")


# ---------------------------------------------------------------------------
# TFB models (deterministic — uses BaseStatistic fallback Normal)
# ---------------------------------------------------------------------------


def run_tfb(series: pd.DataFrame) -> None:
    section("TFB models  (DLinear, PatchTST)  — MC Dropout probabilistic")
    from tsfb.base.models.tfb.adapter import TFBModelAdapter

    for model_key, dropout_note in [
        ("DLinear", "no Dropout → supports_probabilistic=False"),
        ("PatchTST", "has Dropout → supports_probabilistic=True"),
    ]:
        print(f"\n  [{model_key}]  ({dropout_note})")
        try:
            m = TFBModelAdapter(
                model_key,
                input_chunk_length=48,
                output_chunk_length=HORIZON,
                n_epochs=1,
                batch_size=32,
                freq="H",
            )
            m.forecast_fit(train_data=series)
            point = m.forecast(HORIZON, series)
            print(f"    forecast() shape       : {point.shape}")
            print(f"    supports_probabilistic : {m.supports_probabilistic}")

            if m.supports_probabilistic:
                print(f"\n    forecast_probabilistic() — MC Dropout (n={NUM_SAMPLES}):")
                samples = m.forecast_probabilistic(
                    HORIZON, series, num_samples=NUM_SAMPLES
                )
                if samples is not None:
                    show_samples(samples, f"{model_key} MC Dropout samples")
                print(f"\n    forecast_distribution():")
                dist_fc = m.forecast_distribution(HORIZON, series)
                if dist_fc is not None:
                    show_dist(dist_fc, f"{model_key} → Normal")
            else:
                print("    forecast_probabilistic : None  (no Dropout layers)")

        except Exception as exc:
            print(f"    [SKIP] {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Foundation models
# ---------------------------------------------------------------------------


def run_chronos(series: pd.DataFrame, dist_name: str = "normal") -> None:
    section(f"ChronosAdapter  — quantile-based ({dist_name})")
    try:
        from tsfb.base.models.foundation.adapters.chronos_adapter import (
            _CHRONOS_AVAILABLE,
            ChronosAdapter,
        )

        if not _CHRONOS_AVAILABLE:
            print('  [SKIP] pip install "chronos-forecasting>=2.1.0"')
            return
    except ImportError:
        print("  [SKIP] chronos not available")
        return

    # Chronos 2.2.x: use 'amazon/chronos-2' (Chronos2Model architecture)
    # NOT 'amazon/chronos-t5-*' (old T5 architecture, incompatible)
    m = ChronosAdapter(
        pretrained_model_id="amazon/chronos-2",
        device="cpu",
        torch_dtype="float32",
    )
    m.forecast_fit(series)

    print(f"\n  output_type            : {m.output_type}")
    print(f"  supports_probabilistic : {m.supports_probabilistic}")
    print(f"  supports_distribution  : {m.supports_distribution}")

    print("\n  forecast() — point:")
    point = m.forecast(HORIZON, series)
    print(f"    shape={point.shape}  step1={point[0]:.2f}")

    print("\n  forecast_probabilistic() — tiled quantiles:")
    samples = m.forecast_probabilistic(HORIZON, series, num_samples=NUM_SAMPLES)
    show_samples(samples, "Chronos tiled-quantile samples")

    print(f"\n  forecast_distribution('{dist_name}') — quantile-fit:")
    dist_fc = m.forecast_distribution(HORIZON, series, distribution_name=dist_name)
    show_dist(dist_fc, f"Chronos → {dist_name}")

    print("\n  to_payload_dict():")
    payload = dist_fc.to_payload_dict()
    print(f"    keys : {list(payload)}")

    print("\n  to_quantile_dfs([0.1, 0.5, 0.9]):")
    q_dfs = dist_fc.to_quantile_dfs([0.1, 0.5, 0.9], pd.RangeIndex(HORIZON), ["OT"])
    for k, df in q_dfs.items():
        print(f"    {k}: {df.iloc[0,0]:.2f}")


def run_timesfm(series: pd.DataFrame, dist_name: str = "normal") -> None:
    section(f"TimesFMAdapter  — quantile-based ({dist_name})")
    try:
        from tsfb.base.models.foundation.adapters.timesfm_adapter import (
            _TIMESFM_AVAILABLE,
            TimesFMAdapter,
        )

        if not _TIMESFM_AVAILABLE:
            print("  [SKIP] pip install timesfm")
            return
    except ImportError:
        print("  [SKIP] timesfm not available")
        return

    m = TimesFMAdapter(
        pretrained_model_id="google/timesfm-2.5-200m-pytorch",
        device="cpu",
        torch_dtype="float32",
    )
    m.forecast_fit(series)

    print(f"\n  output_type            : {m.output_type}")
    print(f"  supports_probabilistic : {m.supports_probabilistic}")

    print("\n  forecast() — point:")
    point = m.forecast(HORIZON, series)
    print(f"    shape={point.shape}  step1={point[0]:.2f}")

    print("\n  forecast_probabilistic() — tiled quantiles:")
    samples = m.forecast_probabilistic(HORIZON, series, num_samples=NUM_SAMPLES)
    show_samples(samples, "TimesFM tiled-quantile samples")

    print(f"\n  forecast_distribution('{dist_name}'):")
    dist_fc = m.forecast_distribution(HORIZON, series, distribution_name=dist_name)
    show_dist(dist_fc, f"TimesFM → {dist_name}")


def run_moment(series: pd.DataFrame, dist_name: str = "normal") -> None:
    section("MomentAdapter  — MC Dropout (head_dropout=0.1)")
    try:
        from tsfb.base.models.foundation.adapters.moment_adapter import (
            _MOMENT_AVAILABLE,
            MomentAdapter,
        )

        if not _MOMENT_AVAILABLE:
            print("  [SKIP] pip install momentfm  (see install.sh)")
            return
    except ImportError:
        print("  [SKIP] momentfm not available")
        return

    m = MomentAdapter(
        pretrained_model_id="AutonLab/MOMENT-1-small",
        device="cpu",
        torch_dtype="float32",
    )
    m.forecast_fit(series)

    print(f"\n  supports_probabilistic : {m.supports_probabilistic}")

    print("\n  forecast() — point:")
    point = m.forecast(HORIZON, series)
    print(f"    shape={point.shape}  step1={point[0]:.2f}")

    print(f"\n  forecast_probabilistic() — MC Dropout (n={NUM_SAMPLES}):")
    samples = m.forecast_probabilistic(HORIZON, series, num_samples=NUM_SAMPLES)
    show_samples(samples, "MOMENT MC Dropout samples")

    print(f"\n  forecast_distribution('{dist_name}'):")
    dist_fc = m.forecast_distribution(HORIZON, series, distribution_name=dist_name)
    show_dist(dist_fc, f"MOMENT → {dist_name}")


def run_kronos(series: pd.DataFrame, dist_name: str = "normal") -> None:
    section("KronosAdapter  — financial OHLCV model (temperature-varied sampling)")
    try:
        from tsfb.base.models.foundation.adapters.kronos_adapter import (
            _KRONOS_AVAILABLE,
            KronosAdapter,
        )

        if not _KRONOS_AVAILABLE:
            print("  [SKIP] bash install.sh  (clones + installs NeoQuasar/Kronos)")
            return
    except ImportError:
        print("  [SKIP] Kronos not available")
        return

    # Kronos is LLM-style autoregressive: pred_len steps × 2 forward passes per step.
    # Use small horizon + few samples to keep GPU demo fast.
    _kronos_horizon = 4
    _kronos_samples = 5
    print(
        f"  Note: horizon={_kronos_horizon} ({_kronos_horizon * 2} forward passes/call), "
        f"samples={_kronos_samples}"
    )

    try:
        m = KronosAdapter(
            pretrained_model_id="NeoQuasar/Kronos-base",
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
            device="cuda",
            torch_dtype="bfloat16",
            context_length=64,
            temperature=1.0,
        )
        m.forecast_fit(series)
        m._load_model()  # trigger early so we catch download errors
    except Exception as exc:
        print(f"  [SKIP] Model load failed: {type(exc).__name__}: {str(exc)[:100]}")
        print(
            '  Run: python -c "from huggingface_hub import snapshot_download; '
            "snapshot_download('NeoQuasar/Kronos-base')\""
        )
        return

    print(f"\n  supports_probabilistic : {m.supports_probabilistic}")

    print(f"\n  forecast() — point (horizon={_kronos_horizon}):")
    point = m.forecast(_kronos_horizon, series)
    print(f"    shape={point.shape}  step1={point[0]:.2f}")

    print(f"\n  forecast_probabilistic() — T∈[0.6,1.4], n={_kronos_samples}:")
    samples = m.forecast_probabilistic(
        _kronos_horizon, series, num_samples=_kronos_samples
    )
    show_samples(samples, "Kronos samples")

    print(f"\n  forecast_distribution('{dist_name}'):")
    dist_fc = m.forecast_distribution(
        _kronos_horizon,
        series,
        distribution_name=dist_name,
        num_samples=_kronos_samples,
    )
    show_dist(dist_fc, f"Kronos → {dist_name}")


def run_moirai(series: pd.DataFrame) -> None:
    section("MoiraiAdapter  — true Monte Carlo samples")
    try:
        from tsfb.base.models.foundation.adapters.moirai_adapter import (
            _MOIRAI_AVAILABLE,
            MoiraiAdapter,
        )

        if not _MOIRAI_AVAILABLE:
            print("  [SKIP] pip install uni2ts  (see install.sh)")
            return
    except ImportError:
        print("  [SKIP] uni2ts not available")
        return

    m = MoiraiAdapter(
        pretrained_model_id="Salesforce/moirai-1.0-R-small",
        device="cpu",
        inference_mode="zeroshot",
        num_samples=50,
    )
    m.forecast_fit(series)

    print(f"\n  supports_probabilistic : {m.supports_probabilistic}")

    print("\n  forecast_probabilistic() — true MC samples:")
    samples = m.forecast_probabilistic(HORIZON, series, num_samples=NUM_SAMPLES)
    show_samples(samples, "Moirai MC samples")

    print("\n  forecast_distribution() — Normal moment-matched from MC samples:")
    dist_fc = m.forecast_distribution(HORIZON, series, num_samples=NUM_SAMPLES)
    show_dist(dist_fc, "Moirai → Normal")


def run_foundation(series: pd.DataFrame, dist_name: str = "normal") -> None:
    run_chronos(series, dist_name)
    run_timesfm(series, dist_name)
    run_moirai(series)
    run_moment(series, dist_name)
    run_kronos(series, dist_name)


# ---------------------------------------------------------------------------
# Distribution registry showcase
# ---------------------------------------------------------------------------


def run_registry() -> None:
    section("DistributionOutput Registry  — direct instantiation")
    from tsfb.base.forecasting import (
        DistributionForecast,
        NegativeBinomialOutput,
        NormalOutput,
        PoissonOutput,
        StudentTOutput,
        get_distribution_output,
    )

    T = HORIZON
    rng = np.random.default_rng(42)

    for name, out, params in [
        ("Normal(0,1)", NormalOutput(), {"loc": np.zeros(T), "scale": np.ones(T)}),
        (
            "StudentT(df=5,0,1)",
            StudentTOutput(),
            {"df": np.full(T, 5.0), "loc": np.zeros(T), "scale": np.ones(T)},
        ),
        ("Poisson(mu=10)", PoissonOutput(), {"mu": np.full(T, 10.0)}),
        (
            "NegBinomial(5,0.5)",
            NegativeBinomialOutput(),
            {"mu": np.full(T, 5.0), "alpha": np.full(T, 0.5)},
        ),
    ]:
        df_fc = out.distribution(params)
        print(f"\n  {name}")
        print(
            f"    mean={df_fc.mean[0]:.2f}  std={float(np.sqrt(df_fc.variance[0])):.2f}"
            f"  q10={df_fc.quantile(0.1)[0]:.2f}  q90={df_fc.quantile(0.9)[0]:.2f}"
        )

    print("\n  from_samples() — moment-matching:")
    samples = rng.normal(loc=10, scale=2, size=(500, T)).astype(np.float32)
    df_normal = DistributionForecast.from_samples(samples, "normal")
    print(
        f"    fitted  loc={df_normal.params['loc'][0]:.3f}  "
        f"scale={df_normal.params['scale'][0]:.3f}  (true: 10, 2)"
    )

    df_student = DistributionForecast.from_samples(samples, "student_t")
    print(
        f"    student df={df_student.params['df'][0]:.1f}  "
        f"loc={df_student.params['loc'][0]:.3f}"
    )

    print("\n  get_distribution_output('negative_binomial'):")
    out = get_distribution_output("negative_binomial")
    print(f"    params: {out.param_names}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

RUNNERS = {
    # Statistic models
    "poisson": ("PoissonGLM", lambda s, _: run_poisson(s)),
    "negbinom": ("NegativeBinomialGLM", lambda s, _: run_negbinom(s)),
    "prophet": ("ProphetStatistic", lambda s, _: run_prophet(s)),
    # Darts models
    "darts": ("Darts models", lambda s, _: run_darts(s)),
    # TFB deep learning
    "tfb": ("TFB models", lambda s, _: run_tfb(s)),
    # Foundation models (individual)
    "chronos": ("ChronosAdapter", lambda s, d: run_chronos(s, d)),
    "timesfm": ("TimesFMAdapter", lambda s, d: run_timesfm(s, d)),
    "moirai": ("MoiraiAdapter", lambda s, _: run_moirai(s)),
    "moment": ("MomentAdapter", lambda s, d: run_moment(s, d)),
    "kronos": ("KronosAdapter", lambda s, d: run_kronos(s, d)),
    # Convenience groups
    "foundation": ("All foundation models", lambda s, d: run_foundation(s, d)),
    "registry": ("Distribution registry", lambda s, _: run_registry()),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probabilistic forecast demo across model families."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all"] + list(RUNNERS),
        help="Model group to run (default: all)",
    )
    parser.add_argument(
        "--dist",
        default="normal",
        choices=["normal", "student_t"],
        help="Distribution family for Chronos/TimesFM (default: normal)",
    )
    parser.add_argument(
        "--col",
        default="OT",
        help="ETTh1 column to use (default: OT)",
    )
    args = parser.parse_args()

    print(f"\nLoading ETTh1['{args.col}']  n={TRAIN_N}")
    series = load_series(args.col)
    print(
        f"  {series.index[0]} → {series.index[-1]}"
        f"  mean={series.iloc[:,0].mean():.2f}  std={series.iloc[:,0].std():.2f}"
    )

    targets = list(RUNNERS) if args.model == "all" else [args.model]
    # When "all", run each family group once (skip individual foundation runners)
    _foundation_individual = {"chronos", "timesfm", "moirai", "moment", "kronos"}
    if args.model == "all":
        targets = [k for k in RUNNERS if k not in _foundation_individual]

    for key in targets:
        _, fn = RUNNERS[key]
        fn(series, args.dist)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
