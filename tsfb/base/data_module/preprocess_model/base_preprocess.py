from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from darts import TimeSeries

    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None  # type: ignore[assignment]

from tsfb.base.data_module.preprocess_model.constants import SCALER_MAP
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)

ReturnType = Literal["pandas", "darts"]
NormType = Literal["zscore", "minmax"]

RawDataTuple = Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]


class PreprocessData(ABC):
    """
    Abstract base class for data preprocessing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def fit(
        self,
        data: Any,
    ) -> Any:
        """
        Fit preprocessor on training data and return preprocessed
        train/val/test splits (if applicable).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        data: Any,
    ) -> Any:
        """
        Apply already-fitted preprocessing (inference-time transform)
        on new data (no fitting inside).
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply inverse transformation on predictions.
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_time_series(values: pd.Series) -> pd.Series:
        """Convert timestamps to timezone-naive datetimes for downstream models."""
        normalized = pd.to_datetime(values)
        if getattr(normalized.dt, "tz", None) is not None:
            normalized = normalized.dt.tz_convert(None)
        return normalized


class DefaultPreprocessData(PreprocessData):
    """
    Default implementation of the PreprocessData class.

    API:
    - fit(data): used for training
        + split train/val/test
        + fit scalers (if normalize=True)
        + transform the splits
    - predict(data): used for inference
        + do not split
        + only transform full series using scalers fitted/loaded from training
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        cov = config.get("covariate", {}) or {}
        self.past_cols = cov.get("past", [])
        self.future_cols = cov.get("future", [])
        self.static_cols = cov.get("static", [])

        # scalers
        self.target_scalers: Optional[Any] = None
        self.past_scalers: Optional[Any] = None
        self.future_scalers: Optional[Any] = None
        self.static_scalers: Optional[Any] = None

        # used in train mode to fit scalers
        self.train_data: Dict[str, Optional[pd.DataFrame]] = {
            "target": None,
            "past": None,
            "future": None,
            "static": None,
        }

    def _create_and_fit_scaler(self, df: Optional[pd.DataFrame]) -> Optional[Any]:
        """
        Create and fit a scaler for numerical data.

        Args:
            df: DataFrame to fit the scaler on, can be None

        Returns:
            Fitted scaler or None if df is None/empty or has no numerical columns
        """
        if df is not None and not df.empty:
            df_num = df.select_dtypes(include="number")
            if df_num.empty:
                return None
            scaler = self._create_scaler()
            scaler.fit(df_num)
            return scaler
        return None

    def _fit_scalers_with_train_data(self) -> None:
        """Fit scalers using training data."""
        self.target_scalers = self._create_and_fit_scaler(self.train_data["target"])
        self.past_scalers = self._create_and_fit_scaler(self.train_data["past"])
        self.future_scalers = self._create_and_fit_scaler(self.train_data["future"])
        self.static_scalers = self._create_and_fit_scaler(self.train_data["static"])

    def _create_scaler(self) -> Any:
        # normalize may be a bool or a dict, use get(..., {}) for safety
        normalize_cfg = self.config.get("normalize", {}) or {}
        if isinstance(normalize_cfg, dict):
            method = normalize_cfg.get("method", "zscore").lower()
        else:
            # if only True/False provided, fallback to zscore
            method = "zscore"
        scaler_cls = SCALER_MAP.get(method, StandardScaler)
        return scaler_cls()

    def _apply_scaler(self, df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        if scaler is None or df.empty:
            return df
        return pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index,
        )

    def _check_and_fix_duplicate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.duplicated().any():
            dups = df.index[df.index.duplicated()].unique()
            logger.warning("Duplicate timestamps found: %s", dups.tolist())
            return df.groupby(df.index).mean()
        return df

    def _ensure_datetime_index(
        self,
        df: pd.DataFrame,
        freq: Optional[str] = None,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if freq:
            freq = freq.lower()
            df = df.asfreq(freq)
        elif df.index.inferred_freq is not None:
            df = df.asfreq(df.index.inferred_freq.lower())
        else:
            inferred = pd.infer_freq(df.index)
            if inferred:
                df = df.asfreq(inferred.lower())
        return df

    # pylint: disable=too-many-locals
    def _split(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame into train/val/test using one of the modes:
            ratio, last, threshold
        """
        mode = cfg.get("split_mode", "ratio")
        if mode == "ratio":
            ratios = cfg.get("split_ratio", {"train": 0.7, "val": 0.1, "test": 0.2})
            n = len(df)
            tr_end = int(n * ratios.get("train", 0.7))
            val_end = tr_end + int(n * ratios.get("val", 0.1))
            return {
                "train": df.iloc[:tr_end],
                "val": df.iloc[tr_end:val_end],
                "test": df.iloc[val_end:],
            }
        if mode == "last":
            last = cfg.get("split_last", {"val": 0, "test": 0})
            val_n = last.get("val", 0)
            test_n = last.get("test", 0)
            n = len(df)
            if val_n + test_n > n:
                raise ValueError("split_last: val+test exceeds data length")
            return {
                "train": df.iloc[: n - val_n - test_n],
                "val": df.iloc[n - val_n - test_n : n - test_n],
                "test": df.iloc[n - test_n :],
            }

        if mode == "threshold":
            thr = cfg.get("split_threshold", {})
            t_train, t_val, t_test = thr.get("train"), thr.get("val"), thr.get("test")
            if not isinstance(t_train, (list, tuple)) or len(t_train) != 2:
                raise ValueError(
                    "split_threshold requires train=[start_date, end_date]"
                )
            # if not isinstance(t_val, (list, tuple)) or len(t_val) != 2:
            #     raise ValueError(
            #         "split_threshold requires val=[start_date, end_date]"
            #     )

            df_idx = df.index

            def get_mask(t_range):
                start = pd.to_datetime(t_range[0])
                end = pd.to_datetime(t_range[1])
                return (df_idx >= start) & (df_idx <= end)

            train = get_mask(t_train)

            res = {
                "train": df.loc[train],
            }

            # val is optional
            if isinstance(t_val, (list, tuple)) and len(t_val) == 2:
                val = get_mask(t_val)
                res["val"] = df.loc[val]
                boundary_end = pd.to_datetime(t_val[1])
            else:
                res["val"] = pd.DataFrame(columns=df.columns)
                boundary_end = pd.to_datetime(t_train[1])

            if t_test is not None:
                if isinstance(t_test, (list, tuple)) and len(t_test) == 2:
                    test = get_mask(t_test)
                    res["test"] = df.loc[test]
                else:
                    raise ValueError(
                        "split_threshold requires test=[start_date, end_date]"
                    )
            else:
                test = df_idx > boundary_end
                res["test"] = df.loc[test]

            return res

        raise ValueError(f"Invalid split_mode: {mode}")

    def _to_darts(self, df: pd.DataFrame) -> "TimeSeries":
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library is not installed.")
        return TimeSeries.from_dataframe(df, value_cols=list(df.columns))

    def _convert_all_to_darts(
        self,
        splits: Dict[str, pd.DataFrame],
    ) -> Dict[str, "TimeSeries"]:
        return {k: self._to_darts(v) for k, v in splits.items()}

    def fit(
        self,
        data: RawDataTuple,
    ) -> Tuple[
        Dict[str, Union[pd.DataFrame, "TimeSeries"]],
        Dict[str, Dict[str, Union[pd.DataFrame, "TimeSeries"]]],
        Optional[Union[pd.DataFrame, "TimeSeries"]],
    ]:
        """
        Fit preprocessor & preprocess training data.

        Args:
            data: Tuple (target_df, past_df, future_df, static_df)
                  usually the output of DataReader.load_data()

        Returns:
            Tuple:
                - splits_tgt: dict split name -> target data (DataFrame or TimeSeries)
                - splits_cov: dict {"past_covariates": {split->df/ts},
                                    "future_covariates": {split->df/ts}}
                - static_df_ts: optional static features (DataFrame or TimeSeries)
        """
        target_df, past_df, future_df, static_df = data
        freq = self.config.get("data_freq", "H")

        # normalize index
        target_df = self._check_and_fix_duplicate_index(
            self._ensure_datetime_index(target_df, freq)
        )
        if isinstance(past_df, pd.DataFrame):
            past_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(past_df, freq)
            )
        if isinstance(future_df, pd.DataFrame):
            future_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(future_df, freq)
            )

        return self._preprocess_train(
            target_df=target_df,
            past_df=past_df,
            future_df=future_df,
            static_df=static_df,
        )

    def predict(
        self,
        data: RawDataTuple,
    ) -> Tuple[
        Union[pd.DataFrame, "TimeSeries"],
        Dict[str, Optional[Union[pd.DataFrame, "TimeSeries"]]],
        Optional[Union[pd.DataFrame, "TimeSeries"]],
    ]:
        """
        Apply already-fitted preprocessing in inference mode.

        Args:
            data: Tuple (target_df, past_df, future_df, static_df)
                  usually new data for inference.

        Returns:
            Tuple:
                - target_processed: full series (not split) as DataFrame/TimeSeries
                - covariates_processed: {"past_covariates": df/ts or None,
                                         "future_covariates": df/ts or None}
                - static_processed: full static features (DataFrame/TimeSeries) or None
        """
        if isinstance(data, pd.DataFrame):
            target_df = data
            past_df = future_df = static_df = None
        elif isinstance(data, tuple):
            target_df = data[0] if len(data) > 0 else None
            past_df = data[1] if len(data) > 1 else None
            future_df = data[2] if len(data) > 2 else None
            static_df = data[3] if len(data) > 3 else None
        else:
            raise ValueError(f"Unsupported data type for predict: {type(data)}")
        freq = self.config.get("data_freq", "H")

        target_df = self._check_and_fix_duplicate_index(
            self._ensure_datetime_index(target_df, freq)
        )
        if isinstance(past_df, pd.DataFrame):
            past_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(past_df, freq)
            )
        if isinstance(future_df, pd.DataFrame):
            future_df = self._check_and_fix_duplicate_index(
                self._ensure_datetime_index(future_df, freq)
            )

        return self._preprocess_inference(
            target_df=target_df,
            past_df=past_df,
            future_df=future_df,
            static_df=static_df,
        )

    def _preprocess_train(
        self,
        *,
        target_df: pd.DataFrame,
        past_df: Optional[pd.DataFrame],
        future_df: Optional[pd.DataFrame],
        static_df: Optional[pd.DataFrame],
    ) -> Tuple[
        Dict[str, Union[pd.DataFrame, "TimeSeries"]],
        Dict[str, Dict[str, Union[pd.DataFrame, "TimeSeries"]]],
        Optional[Union[pd.DataFrame, "TimeSeries"]],
    ]:
        """Preprocess in train mode: split + fit scaler + transform splits."""
        splits_tgt = self._split(target_df, self.config)
        splits_past = {k: pd.DataFrame() for k in ("train", "val", "test")}
        splits_future = {k: pd.DataFrame() for k in ("train", "val", "test")}

        if isinstance(past_df, pd.DataFrame):
            splits_past.update(self._split(past_df, self.config))
        if isinstance(future_df, pd.DataFrame):
            # future covariates are typically known in advance -> copy to all splits
            for part in ("train", "val", "test"):
                splits_future[part] = future_df.copy()

        # prepare train_data to fit scaler
        self.train_data = {
            "target": splits_tgt["train"],
            "past": splits_past["train"] if isinstance(past_df, pd.DataFrame) else None,
            "future": (
                splits_future["train"] if isinstance(future_df, pd.DataFrame) else None
            ),
            "static": static_df if isinstance(static_df, pd.DataFrame) else None,
        }

        if self.config.get("normalize"):
            # fit scaler
            self._fit_scalers_with_train_data()

            # transform all splits + static
            for part in ("train", "val", "test"):
                splits_tgt[part] = self._apply_scaler(
                    splits_tgt[part], self.target_scalers
                )
                splits_past[part] = self._apply_scaler(
                    splits_past[part], self.past_scalers
                )
                splits_future[part] = self._apply_scaler(
                    splits_future[part], self.future_scalers
                )

            if isinstance(static_df, pd.DataFrame):
                static_df = self._apply_scaler(static_df, self.static_scalers)

        # convert to darts if needed
        if self.config.get("return_type") == "darts":
            splits_tgt = self._convert_all_to_darts(splits_tgt)
            splits_past = self._convert_all_to_darts(splits_past)
            splits_future = self._convert_all_to_darts(splits_future)
            if isinstance(static_df, pd.DataFrame):
                static_df = self._to_darts(static_df)

        splits_cov = {
            "past_covariates": splits_past,
            "future_covariates": splits_future,
        }

        return splits_tgt, splits_cov, static_df

    def _preprocess_inference(
        self,
        *,
        target_df: pd.DataFrame,
        past_df: Optional[pd.DataFrame],
        future_df: Optional[pd.DataFrame],
        static_df: Optional[pd.DataFrame],
    ) -> Tuple[
        Union[pd.DataFrame, "TimeSeries"],
        Dict[str, Optional[Union[pd.DataFrame, "TimeSeries"]]],
        Optional[Union[pd.DataFrame, "TimeSeries"]],
    ]:
        """
        Preprocess in inference mode:
        - Do not split train/val/test
        - only apply scaler (if configured) on the full series
        - requires scalers to be pre-fitted (from training) if normalize=True
        """
        if self.config.get("normalize"):
            if (
                self.target_scalers is None
                and self.past_scalers is None
                and self.future_scalers is None
                and self.static_scalers is None
            ):
                raise RuntimeError(
                    "Scalers are not fitted. "
                    "Call fit() first, or assign scalers loaded from training "
                    "before calling predict()."
                )

            target_df = self._apply_scaler(target_df, self.target_scalers)
            if isinstance(past_df, pd.DataFrame):
                past_df = self._apply_scaler(past_df, self.past_scalers)
            if isinstance(future_df, pd.DataFrame):
                future_df = self._apply_scaler(future_df, self.future_scalers)
            if isinstance(static_df, pd.DataFrame):
                static_df = self._apply_scaler(static_df, self.static_scalers)

        # convert to darts if needed
        if self.config.get("return_type") == "darts":
            target_ts: Union[pd.DataFrame, "TimeSeries"] = self._to_darts(target_df)
            past_ts: Optional[Union[pd.DataFrame, "TimeSeries"]] = (
                self._to_darts(past_df) if isinstance(past_df, pd.DataFrame) else None
            )
            future_ts: Optional[Union[pd.DataFrame, "TimeSeries"]] = (
                self._to_darts(future_df)
                if isinstance(future_df, pd.DataFrame)
                else None
            )
            static_ts: Optional[Union[pd.DataFrame, "TimeSeries"]] = (
                self._to_darts(static_df)
                if isinstance(static_df, pd.DataFrame)
                else None
            )
        else:
            target_ts = target_df
            past_ts = past_df
            future_ts = future_df
            static_ts = static_df

        covariates = {
            "past_covariates": past_ts,
            "future_covariates": future_ts,
        }

        return target_ts, covariates, static_ts

    def inverse_predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        target_df = data.copy() if isinstance(data, pd.DataFrame) else data
        if self.config.get("normalize") and self.target_scalers is not None:
            if isinstance(target_df, pd.DataFrame):
                scaler = self.target_scalers
                feature_cols = [
                    c
                    for c in target_df.columns
                    if c in self.config.get("target_columns", [])
                ]
                if not feature_cols:
                    feature_cols = list(target_df.columns)
                num_features = getattr(scaler, "n_features_in_", len(feature_cols))
                dummy = np.zeros((len(target_df), num_features))
                dummy[:, : len(feature_cols)] = target_df[feature_cols].values
                target_df[feature_cols] = scaler.inverse_transform(dummy)[
                    :, : len(feature_cols)
                ]
        return target_df
