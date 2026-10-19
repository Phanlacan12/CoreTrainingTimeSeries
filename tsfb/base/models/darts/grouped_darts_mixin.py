from typing import Any, Dict, List, Optional

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.models.forecasting.arima import ARIMA as DartsARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX as staARIMA

from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class GroupedDartsMixin:
    """Shared helpers for grouped Darts fit/predict data conversion."""

    @staticmethod
    def _normalize_group_cols(group_cols: Optional[List[str] | str]) -> List[str]:
        if group_cols is None:
            return []
        if isinstance(group_cols, str):
            return [group_cols]
        return list(group_cols)

    @staticmethod
    def _group_ids_from_frame(
        frame: pd.DataFrame,
        group_cols: List[str],
    ) -> List[Any]:
        unique_groups = frame[group_cols].drop_duplicates().sort_values(group_cols)
        if len(group_cols) == 1:
            return unique_groups[group_cols[0]].tolist()
        return [tuple(row) for row in unique_groups.to_numpy().tolist()]

    def _merge_static_into_grouped(
        self,
        frame: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]],
        group_cols: List[str],
        static_cols: Optional[List[str]],
    ) -> pd.DataFrame:
        if not static_cols:
            return frame

        static_df = covariates.get("static_covariates") if covariates else None
        if not isinstance(static_df, pd.DataFrame):
            return frame

        missing_group_cols = [c for c in group_cols if c not in static_df.columns]
        if missing_group_cols:
            raise ValueError(
                "Grouped static_covariates is missing group columns: "
                f"{missing_group_cols}"
            )

        wanted_static_cols = [c for c in static_cols if c in static_df.columns]
        if not wanted_static_cols:
            return frame

        deduped = static_df[group_cols + wanted_static_cols].drop_duplicates(
            subset=group_cols
        )
        merged = frame.drop(columns=wanted_static_cols, errors="ignore").merge(
            deduped,
            on=group_cols,
            how="left",
        )
        return merged

    # pylint: disable=too-many-arguments
    def _grouped_frame_to_timeseries(
        self,
        frame: Optional[pd.DataFrame],
        *,
        group_cols: List[str],
        time_col: str,
        value_cols: Optional[List[str]] = None,
        static_cols: Optional[List[str]] = None,
    ) -> Optional[List[TimeSeries]]:
        """Convert a grouped dataframe into one Darts series per group."""
        if frame is None or not isinstance(frame, pd.DataFrame):
            return None

        if time_col not in frame.columns or not all(
            c in frame.columns for c in group_cols
        ):
            return None

        value_cols = value_cols or [
            c
            for c in frame.columns
            if c not in set(group_cols + [time_col] + list(static_cols or []))
        ]
        if not value_cols:
            return None

        sorted_frame = frame.copy()
        sorted_frame[time_col] = pd.to_datetime(sorted_frame[time_col])
        sorted_frame = sorted_frame.sort_values(group_cols + [time_col]).reset_index(
            drop=True
        )

        return TimeSeries.from_group_dataframe(
            sorted_frame,
            time_col=time_col,
            group_cols=group_cols,
            value_cols=value_cols,
            static_cols=static_cols,
        )

    def _encode_grouped_static_covariates_if_needed(
        self, ts_list: List[TimeSeries]
    ) -> List[TimeSeries]:
        if not ts_list or ts_list[0].static_covariates is None:
            return ts_list
        dtypes = ts_list[0].static_covariates.dtypes.to_dict()
        logger.info(
            "[AutoEncode] Grouped static covariates dtype before encode: %s", dtypes
        )
        try:
            transformer = StaticCovariatesTransformer()
            ts_list = transformer.fit_transform(ts_list)
            self._static_transformer = transformer
            logger.info("[AutoEncode] Grouped static covariates encoded successfully.")
            return ts_list
        except Exception as exc:
            logger.warning(
                "[AutoEncode] Grouped static encoding failed: %s. "
                "Dropping static covariates.",
                exc,
            )
            self._static_transformer = None
            for ts in ts_list:
                ts._static_covariates = None  # pylint: disable=protected-access
            return ts_list

    def _transform_grouped_static_covariates_if_needed(
        self, ts_list: List[TimeSeries]
    ) -> List[TimeSeries]:
        if not ts_list or ts_list[0].static_covariates is None:
            return ts_list

        if self._static_transformer is not None:
            try:
                return self._static_transformer.transform(ts_list)
            except Exception as exc:
                logger.warning(
                    "[AutoEncode] Reusing grouped static transformer failed: %s. "
                    "Re-fitting grouped static transformer.",
                    exc,
                )

        return self._encode_grouped_static_covariates_if_needed(ts_list)

    # pylint: disable=too-many-arguments
    def _grouped_predictions_to_frame(
        self,
        preds_ts: Any,
        group_ids: List[Any],
        *,
        group_cols: List[str],
        time_col: str,
        target_cols: List[str],
    ) -> pd.DataFrame:
        """Flatten grouped Darts predictions back into a single dataframe."""
        ts_list = preds_ts if isinstance(preds_ts, list) else [preds_ts]
        if len(ts_list) != len(group_ids):
            raise ValueError(
                "Grouped prediction count does not match group count: "
                f"{len(ts_list)} != {len(group_ids)}"
            )

        frames = []
        for group_id, pred_ts in zip(group_ids, ts_list):
            pred_df = pred_ts.to_dataframe(time_as_index=False)
            if time_col not in pred_df.columns:
                pred_df = pred_df.rename(columns={pred_df.columns[0]: time_col})
            value_cols = [c for c in pred_df.columns if c != time_col]
            if (
                len(target_cols) == 1
                and value_cols
                and target_cols[0] not in pred_df.columns
            ):
                pred_df = pred_df.rename(columns={value_cols[0]: target_cols[0]})

            if len(group_cols) == 1:
                pred_df[group_cols[0]] = group_id
            else:
                for col_name, col_value in zip(group_cols, group_id):
                    pred_df[col_name] = col_value

            frames.append(pred_df[[time_col] + group_cols + target_cols])

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(group_cols + [time_col])
            .reset_index(drop=True)
        )


class PatchedARIMA(DartsARIMA):
    """ARIMA wrapper that forwards ``measurement_error`` to statsmodels."""

    def __init__(self, *args, measurement_error=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement_error = measurement_error
        self.training_historic_future_covariates = None

    def _fit(self, series, future_covariates=None):
        super(DartsARIMA, self)._fit(series, future_covariates)
        self._assert_univariate(series)
        self.training_historic_future_covariates = future_covariates
        model = staARIMA(
            series.values(copy=False),
            exog=future_covariates.values(copy=False) if future_covariates else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            measurement_error=self.measurement_error,
        )
        self.model = model.fit(low_memory=True)
        return self
