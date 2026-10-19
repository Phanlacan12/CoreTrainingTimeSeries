from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach, GroupedForecastingApproach
from tsfb.base.approaches.default import DefaultApproach
from tsfb.base.approaches.univariate import UnivariateToMultivariate
from tsfb.base.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from tsfb.base.models.base_model import BatchMaker
from tsfb.base.models.model_loader import get_single_model


class HybridApproach(GroupedForecastingApproach):
    """Route different sub-approaches to different subsets of series."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = config
        self.feature_names: List[str] = []
        self.models: Dict[str, Any] = {}

    @staticmethod
    def _group_values_for_block(block: Dict[str, Any]) -> List[Any]:
        values = block.get("group_values")
        if values is None:
            values = block.get("series")
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ValueError(
                "Hybrid grouped configuration requires 'group_values' or 'series' "
                "to be a list of group ids."
            )
        return list(values)

    def _filter_grouped_covariates(
        self,
        covariates: Optional[Dict[str, pd.DataFrame]],
        group_values: List[Any],
    ) -> Optional[Dict[str, pd.DataFrame]]:
        if not covariates or self.group_col is None:
            return covariates

        out: Dict[str, pd.DataFrame] = {}
        for name, frame in covariates.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            if self.group_col in frame.columns:
                out[name] = frame[frame[self.group_col].isin(group_values)].copy()
            else:
                out[name] = frame.copy()
        return out

    def _validate_group_assignments(self, available_groups: List[Any]) -> None:
        seen: set[Any] = set()
        available = set(available_groups)
        for block in self.config["group"]:
            values = self._group_values_for_block(block)
            overlap = seen.intersection(values)
            if overlap:
                raise ValueError(
                    "Hybrid grouped configuration contains overlapping group ids: "
                    f"{sorted(overlap)}"
                )
            seen.update(values)

        missing = available - seen
        extra = seen - available
        if missing:
            raise ValueError(
                "Hybrid grouped configuration does not cover all dataset groups. "
                f"Missing: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                "Hybrid grouped configuration references unknown group ids. "
                f"Unknown: {sorted(extra)}"
            )

    def _fit_grouped(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]],
        **kwargs: Any,
    ) -> "HybridApproach":
        assert self.group_col is not None
        assert self.time_col is not None

        available_groups = (
            self._sort_grouped_frame(train_data, self.group_col, self.time_col)[
                self.group_col
            ]
            .drop_duplicates()
            .tolist()
        )
        self._validate_group_assignments(available_groups)
        self.models = {}

        for grp in self.config["group"]:
            group_values = self._group_values_for_block(grp)
            sub = grp["sub_approach"]

            if sub not in ["univariate_per_series", "multivariate"]:
                raise ValueError(f"Unknown sub_approach: {sub}")

            factory = partial(
                get_single_model,
                grp["model_name"],
                grp.get("model_hyper_params", {}),
            )
            approach: ForecastingApproach
            if sub == "univariate_per_series":
                approach = UnivariateToMultivariate(model_factory=factory)
            else:
                approach = DefaultApproach(model_factory=factory)

            subset_df = train_data[train_data[self.group_col].isin(group_values)].copy()
            subset_cov = self._filter_grouped_covariates(covariates, group_values)
            approach.forecast_fit(subset_df, covariates=subset_cov, **kwargs)
            self.models[",".join(map(str, group_values))] = (approach, group_values)

        return self

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "HybridApproach":
        if "group" not in self.config:
            raise KeyError("Required key 'group' not found in config")

        if self._configure_grouped_mode(train_data, kwargs):
            return self._fit_grouped(train_data, covariates, **kwargs)

        self.feature_names = train_data.columns.tolist()

        for grp in self.config["group"]:
            cols = grp["series"]
            sub = grp["sub_approach"]

            if sub not in ["univariate_per_series", "multivariate"]:
                raise ValueError(f"Unknown sub_approach: {sub}")

            factory = partial(
                get_single_model,
                grp["model_name"],
                grp.get("model_hyper_params", {}),
            )

            approach: ForecastingApproach
            if sub == "univariate_per_series":
                approach = UnivariateToMultivariate(model_factory=factory)
            else:
                approach = DefaultApproach(model_factory=factory)

            subset_df = train_data[cols]
            approach.forecast_fit(subset_df, covariates=covariates, **kwargs)
            self.models[",".join(cols)] = (approach, cols)

        return self

    def _forecast_grouped(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Must call forecast_fit() before forecast()")
        assert self.group_col is not None
        assert self.time_col is not None

        outputs = []
        for approach, group_values in self.models.values():
            subset_df = lookback_data[
                lookback_data[self.group_col].isin(group_values)
            ].copy()
            subset_cov = self._filter_grouped_covariates(covariates, group_values)
            outputs.append(
                approach.forecast(
                    horizon,
                    subset_df,
                    covariates=subset_cov,
                    **kwargs,
                )
            )

        return self._finalize_grouped_outputs(outputs)

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if self.grouped_mode:
            return self._forecast_grouped(
                horizon,
                lookback_data,
                covariates=covariates,
                **kwargs,
            )

        if not self.models:
            raise RuntimeError("Must call forecast_fit() before forecast()")

        outputs = []
        for approach, cols in self.models.values():
            outputs.append(
                approach.forecast(
                    horizon, lookback_data[cols], covariates=covariates, **kwargs
                )
            )
        return pd.concat(outputs, axis=1)

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> np.ndarray:
        if self.grouped_mode:
            raise NotImplementedError(
                "Grouped batch_forecast is not supported for HybridApproach. "
                "Use grouped sample-based rolling evaluation instead."
            )

        if not self.models:
            raise RuntimeError("Must call forecast_fit() before batch_forecast()")

        inner = getattr(batch_maker, "_batch_maker", None)
        if inner is None:
            raise ValueError("batch_maker must wrap RollingForecastEvalBatchMaker")

        if isinstance(inner.series, np.ndarray):
            full_df = getattr(
                inner,
                "series_df",
                pd.DataFrame(inner.series, columns=self.feature_names),
            )
        else:
            full_df = inner.series

        all_arrs: List[np.ndarray] = []

        for approach, target_cols in self.models.values():
            sub_series = full_df[target_cols]

            sub_eval = RollingForecastEvalBatchMaker(
                series=sub_series,
                index_list=inner.index_list,
                covariates=inner.covariates,
            )

            sub_pred = RollingForecastPredictBatchMaker(sub_eval)
            arr = approach.batch_forecast(horizon, sub_pred, **kwargs)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            all_arrs.append(arr)

        return np.concatenate(all_arrs, axis=2)  # (n_windows, horizon, total_features)
