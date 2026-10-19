"""
KronosAdapter — wraps KronosPredictor (shiyu-coder/Kronos).

NOTE: Kronos is a financial OHLCV time series model. For general univariate
series, input values are mapped to the 'close' price column; open/high/low
are set to the same value, volume/amount to 0.

Probabilistic forecasting: call predict() N times with slight temperature
variation (T ∈ [0.8, 1.2]) to get diverse samples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tsfb.base.forecasting.distribution_forecast import DistributionForecast
from tsfb.base.models.foundation.base_foundation import FoundationModelBase
from tsfb.base.schema.foundation_config import FineTuneConfig

try:
    from kronos import Kronos, KronosPredictor, KronosTokenizer

    _KRONOS_AVAILABLE = True
except ImportError:
    _KRONOS_AVAILABLE = False


class KronosAdapter(FoundationModelBase):  # pylint: disable=abstract-method
    """
    Adapter for Kronos (shiyu-coder/Kronos) foundation model.

    Kronos is an autoregressive model designed for financial OHLCV data.
    For general univariate series the input values are treated as 'close' price.
    Sampling is done via temperature variation (autoregressive LLM-style).

    Required extra dependency:
        bash install.sh   (clones + patches Kronos into vendor/Kronos/)
    """

    # OHLCV column names expected by KronosPredictor
    _PRICE_COLS = ["open", "high", "low", "close"]
    _VOL_COL = "volume"
    _AMT_COL = "amount"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pretrained_model_id: str = "NeoQuasar/Kronos-base",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cuda",
        inference_mode: str = "zeroshot",
        context_length: int = 512,
        torch_dtype: str = "bfloat16",
        finetune_config: Optional[dict] = None,
        temperature: float = 1.0,
        **_kwargs,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            finetune_config=finetune_config,
            pretrained_model_id=pretrained_model_id,
            inference_mode=inference_mode,
            context_length=context_length,
        )
        self._tokenizer_id = tokenizer_id
        self._temperature = temperature
        self._predictor: Optional[KronosPredictor] = None

    @property
    def model_name(self) -> str:
        return f"kronos[{self._model_id}]"

    @property
    def supports_finetuning(self) -> bool:
        return False  # KronosPredictor has no fine-tuning API

    @property
    def supports_probabilistic(self) -> bool:
        return True

    @staticmethod
    def _resolve_path(repo_id: str) -> str:
        """Return local cache path when available, otherwise fall back to repo_id.

        Search order:
        1. ~/.cache/kronos_manual/models--<slug>/          (download_kronos.py output)
        2. HuggingFace hub snapshot dir (latest snapshot with model.safetensors)
        3. repo_id string → HuggingFace download
        """
        slug = repo_id.replace("/", "--")

        # 1. Manual download cache
        manual = Path.home() / ".cache" / "kronos_manual" / f"models--{slug}"
        if (manual / "model.safetensors").exists():
            return str(manual)

        # 2. HuggingFace hub snapshot (picks latest snapshot dir)
        snapshots = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{slug}"
            / "snapshots"
        )
        if snapshots.is_dir():
            candidates = sorted(
                d
                for d in snapshots.iterdir()
                if d.is_dir() and (d / "model.safetensors").exists()
            )
            if candidates:
                return str(candidates[-1])

        # 3. Fall back to HuggingFace download
        return repo_id

    def _load_model(self) -> None:
        if self._predictor is not None:
            return
        if not _KRONOS_AVAILABLE:
            raise ImportError(
                "Kronos is required for KronosAdapter.\n"
                "Install with: bash install.sh  "
                "(clones + patches shiyu-coder/Kronos into vendor/Kronos/)"
            )
        tok_path = self._resolve_path(self._tokenizer_id)
        model_path = self._resolve_path(self._model_id)
        tokenizer = KronosTokenizer.from_pretrained(tok_path)
        model = Kronos.from_pretrained(model_path)
        self._predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=self._device,
            max_context=self._context_length or 512,
        )

    def _series_to_ohlcv(self, series: pd.DataFrame) -> tuple:
        """Map a univariate (or 'close'-column) DataFrame to OHLCV format.

        Returns (ohlcv_df, x_timestamp) ready for KronosPredictor.predict().
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("KronosAdapter requires a DatetimeIndex on the series.")

        # Use first numeric column as the 'close' price
        numeric_cols = series.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError("KronosAdapter: series has no numeric columns.")
        close_vals = series[numeric_cols[0]].to_numpy(dtype=np.float64)

        df = pd.DataFrame(index=series.index)
        df["open"] = close_vals
        df["high"] = close_vals
        df["low"] = close_vals
        df["close"] = close_vals
        df["volume"] = 0.0
        df["amount"] = 0.0

        # Limit to context_length
        ctx_len = self._context_length or 512
        df = df.iloc[-ctx_len:]
        return df, df.index

    def _future_index(self, series: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
        """Generate a future DatetimeIndex for the forecast horizon."""
        idx = series.index
        freq = pd.infer_freq(idx) or "H"
        return pd.date_range(
            start=idx[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        self._load_model()
        ohlcv_df, x_ts = self._series_to_ohlcv(series)
        y_ts = self._future_index(series, horizon)

        pred_df = self._predictor.predict(  # type: ignore[union-attr]
            df=ohlcv_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=self._temperature,
            verbose=False,
        )
        return pred_df["close"].to_numpy(dtype=np.float32)[:horizon]

    def forecast_probabilistic(
        self,
        horizon: int,
        series: pd.DataFrame,
        num_samples: int = 100,
        *,
        covariates: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Generate samples via temperature-varied autoregressive inference.

        Kronos uses LLM-style sampling — different temperatures produce
        different trajectories, approximating a predictive distribution.
        """
        self._load_model()
        ohlcv_df, x_ts = self._series_to_ohlcv(series)
        y_ts = self._future_index(series, horizon)

        # Temperature range: [0.6, 1.4] — wider = more diverse samples
        temperatures = np.linspace(0.6, 1.4, num_samples)
        draws: list = []
        for t in temperatures:
            pred_df = self._predictor.predict(  # type: ignore[union-attr]
                df=ohlcv_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=horizon,
                T=float(t),
                verbose=False,
            )
            draws.append(pred_df["close"].to_numpy(dtype=np.float32)[:horizon])

        return np.stack(draws).astype(np.float32)  # (num_samples, horizon)

    def forecast_distribution(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        series: pd.DataFrame,
        distribution_name: str = "normal",
        num_samples: int = 500,
        *,
        covariates: Optional[dict] = None,
    ) -> DistributionForecast:
        """Fit a parametric distribution to temperature-varied samples."""
        samples = self.forecast_probabilistic(
            horizon, series, num_samples=num_samples, covariates=covariates
        )
        return DistributionForecast.from_samples(samples, distribution_name)

    def finetune(
        self,
        train_data: pd.DataFrame,
        config: "FineTuneConfig",
        val_data: Optional[pd.DataFrame] = None,
    ) -> "KronosAdapter":
        raise NotImplementedError(
            "KronosAdapter does not support fine-tuning via the standard interface. "
            "See vendor/Kronos/finetune/ for fine-tuning scripts."
        )
