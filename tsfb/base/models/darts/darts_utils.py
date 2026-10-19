# pylint: disable=duplicate-code
import contextlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from tsfb.base.models.base_model import BatchMaker
from tsfb.base.schema.darts_config import CovariateSeries, TimeSeriesGroup
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class DartsUtilsMixin:
    """Helper mixin for Darts-related utilities.

    Provides methods to detect grouped DataFrames, convert pandas DataFrames
    to Darts TimeSeries (including covariates), and helpers used by Darts
    model adapters for preparing training/forecasting inputs.
    """

    def __init__(self) -> None:
        self._full_future_cov: Optional[pd.DataFrame] = None
        self.model: Optional[ForecastingModel] = None

    def _is_grouped_frame(
        self, df: pd.DataFrame, group_cols: List[str], time_col: str
    ) -> bool:
        if not isinstance(df, pd.DataFrame):
            return False
        if time_col not in df.columns:
            return False
        if not group_cols or not any(c in df.columns for c in group_cols):
            return False
        return True

    def _extract_forecast_covariates(self, series, horizon, covariates):  # noqa: C901
        """Extract and align past/future covariates for forecasting."""
        past_cov = future_cov = None

        def extend_df(df: pd.DataFrame, target_len: int) -> pd.DataFrame:
            """Repeat last row until DataFrame reaches `target_len`."""
            if df is None or df.empty:
                return df
            steps = target_len - len(df)
            if steps <= 0:
                return df
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex):
                freq = idx.freq or pd.infer_freq(idx) or pd.infer_freq(series.index)
                if not freq:
                    raise ValueError("Cannot infer frequency for datetime covariates.")
                new_idx = pd.date_range(
                    idx[-1] + pd.Timedelta(freq), periods=steps, freq=freq
                )
            elif isinstance(idx, pd.RangeIndex):
                new_idx = pd.RangeIndex(
                    idx[-1] + idx.step, idx[-1] + idx.step * (steps + 1), idx.step
                )
            else:
                step = (
                    idx.to_series().diff().dropna().mode().iloc[0]
                    if len(idx) > 1
                    else 1
                )
                new_idx = np.arange(idx[-1] + step, idx[-1] + step * (steps + 1), step)
            ext = pd.DataFrame(
                np.repeat(df.iloc[[-1]].values, steps, axis=0),
                index=new_idx,
                columns=df.columns,
            )
            return pd.concat([df, ext])

        def make_cov(df: pd.DataFrame, needed_len: int, label: str) -> "TimeSeries":
            if len(df) < needed_len:
                logger.warning(
                    "[CovariateFix] %s too short (%d < %d); extending.",
                    label,
                    len(df),
                    needed_len,
                )
                df = extend_df(df, needed_len)
            return TimeSeries.from_dataframe(df)

        if covariates:
            pc, fc = covariates.get("past_covariates"), covariates.get(
                "future_covariates"
            )
            if pc is not None:
                past_cov = make_cov(pc, len(series), "past_covariates")
            if fc is not None:
                future_cov = make_cov(fc, len(series) + horizon, "future_covariates")
        elif self._full_future_cov is not None:
            fc = self._full_future_cov
            needed_len = len(series) + horizon
            if len(fc) < needed_len:
                fc = extend_df(fc, needed_len)
            delta = series.index.freq or (series.index[1] - series.index[0])
            start = series.index[-1] + delta
            new_idx = pd.date_range(start=start, periods=horizon, freq=delta)
            future_cov = TimeSeries.from_dataframe(fc.reindex(new_idx))

        return past_cov, future_cov

    def _convert_covariates(self, cov_train, cov_val):
        ts_train = (
            TimeSeries.from_dataframe(cov_train) if cov_train is not None else None
        )
        ts_val = TimeSeries.from_dataframe(cov_val) if cov_val is not None else None
        return ts_train, ts_val

    def _convert_main_series(self, train, valid, static_covariates):
        # The DataFrame may contain non-numeric columns
        # (e.g. timestamp strings, category labels)
        # that were not set as the index. Darts requires all value columns
        # to be numeric, so keep only numeric columns here.
        non_numeric = train.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            logger.debug(
                "Dropping non-numeric columns before Darts conversion: %s",
                non_numeric,
            )
            train = train.select_dtypes(include="number")
            if valid is not None:
                valid = valid.select_dtypes(include="number")

        if isinstance(train.index, pd.DatetimeIndex) and train.index.freq is None:
            inferred = pd.infer_freq(train.index)
            if inferred:
                train = train.asfreq(inferred)
        if (
            valid is not None
            and isinstance(valid.index, pd.DatetimeIndex)
            and valid.index.freq is None
        ):
            inferred = pd.infer_freq(valid.index)
            if inferred:
                valid = valid.asfreq(inferred)
        ts_train = TimeSeries.from_dataframe(train, static_covariates=static_covariates)
        ts_valid = (
            TimeSeries.from_dataframe(valid, static_covariates=static_covariates)
            if valid is not None
            else None
        )
        return ts_train, ts_valid

    def _get_batch_data(
        self,
        batch_maker: BatchMaker,
        eval_maker,
        n_windows: int,
        win_size: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.Index]:
        batch = batch_maker.make_batch(batch_size=n_windows, win_size=win_size)
        x_windows = batch["input"]
        cov_dict = batch.get("covariates", {})
        full_idx = getattr(eval_maker, "index", None)
        if full_idx is None:
            full_idx = pd.RangeIndex(len(eval_maker.series))
        return x_windows, cov_dict, full_idx

    @contextlib.contextmanager
    def _suppress_lightning_logs(self):
        """
        Context manager to suppress PyTorch Lightning logs during model fit/predict.
        """
        pl = logging.getLogger("pytorch_lightning")
        old = pl.level
        pl.setLevel(logging.CRITICAL)
        try:
            yield
        finally:
            pl.setLevel(old)

    def _convert_to_timeseries(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame],
        covs: CovariateSeries,
        covariates: Optional[Dict[str, pd.DataFrame]],
    ) -> TimeSeriesGroup:
        """
        Convert data and covariates to Darts TimeSeries objects.
        """
        ts_train, ts_valid = self._convert_main_series(
            train, valid, self._get_static_covariates(covariates)
        )
        pt, pv = self._convert_covariates(covs.past_train, covs.past_val)
        ft, fv = self._convert_covariates(covs.future_train, covs.future_val)
        return TimeSeriesGroup(
            ts_train,
            ts_valid,
            CovariateSeries(pt, pv, ft, fv),
        )

    # Several early exits keep this helper readable because the acceptable
    # return shapes depend on both model support and covariate encoding state.
    # pylint: disable=too-many-return-statements
    def _get_static_covariates(self, covariates):
        """
        Extract valid static covariates for Darts TimeSeries.
        Must return pd.Series, pd.DataFrame (1 row), or None.

        Handles encoded multi-entity DataFrames by stripping identifier
        columns and deduplicating rows.
        """
        if not self._model_supports_static_covariates():
            return None
        sc = covariates.get("static_covariates") if covariates else None
        if sc is None:
            return None

        if isinstance(sc, pd.DataFrame):
            # Drop identifier columns that are not model features
            id_cols = [
                c
                for c in sc.columns
                if c.endswith("_id") or c in ("group_id", "entity_id")
            ]
            if id_cols:
                sc = sc.drop(columns=id_cols, errors="ignore")
            if sc.empty:
                return None

            # Deduplicate rows (static values repeat per timestamp/entity)
            deduped = sc.drop_duplicates()
            if len(deduped) <= 1:
                return deduped.reset_index(drop=True)

            # Multiple distinct rows → single-series limitation: use first
            logger.info(
                "[StaticCov] %d distinct static covariate rows; "
                "using first row for single-series mode.",
                len(deduped),
            )
            return deduped.iloc[[0]].reset_index(drop=True)

        if isinstance(sc, pd.Series):
            return sc
        if isinstance(sc, dict):
            return pd.Series(sc)

        raise TypeError(
            f"Invalid type for static_covariates: {type(sc)}; "
            f"expected pd.Series, pd.DataFrame, or None."
        )

    def _model_supports_static_covariates(self) -> bool:
        """
        Check if the model supports static covariates.
        :return: True if supported, else False.
        """
        return getattr(self.model, "supports_static_covariates", False)
