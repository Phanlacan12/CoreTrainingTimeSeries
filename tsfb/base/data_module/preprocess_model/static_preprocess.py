from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from tsfb.base.data_module.preprocess_model.base_preprocess import PreprocessData
from tsfb.base.data_module.preprocess_model.constants import SCALER_MAP
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class StaticGroupedPreprocess(PreprocessData):
    """
    Data preprocessor for multi-entity data with static covariates.

    API:
      - fit(data):
          * data can be a single DataFrame or a tuple/list of DataFrames
            (target_df, past_cov_df, future_cov_df, static_df)
          * merge raw inputs into one DataFrame
          * optionally normalize
          * store full DataFrame and per-entity DataFrames
      - predict(data):
          * same input structure as fit()
          * merge + transform using already-fitted scalers
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.group_col = config.get("group_col") or config.get("id_col")
        self.time_col = config.get("time_col") or config.get("date_col")
        self.target_cols = config["target_columns"]

        cov = config.get("covariate", {}) or {}
        self.past_cols = cov.get("past", [])
        self.future_cols = cov.get("future", [])
        self.static_cols = cov.get("static", [])

        self.normalize_cfg = config.get("normalize", {}) or {}
        self.normalize = bool(self.normalize_cfg)
        self.norm_method = self.normalize_cfg.get("method", "zscore").lower()
        self.per_entity = self.normalize_cfg.get("per_entity", True)

        # Will be set in fit/predict
        self.df_full: Optional[pd.DataFrame] = None
        self.raw_df_full: Optional[pd.DataFrame] = None
        self.entity_dfs: Dict[Any, pd.DataFrame] = {}

        # scalers
        self.global_scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.entity_scalers: Dict[Any, StandardScaler | MinMaxScaler] = {}

        # Static covariate encoding / scaling
        self._static_label_encoders: Dict[str, LabelEncoder] = {}
        self._static_scaler: Optional[StandardScaler | MinMaxScaler] = None
        self._static_scaler_cols: List[str] = []
        self._encoded_static_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        data: Union[pd.DataFrame, Tuple[pd.DataFrame, ...], List[pd.DataFrame]],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Union[Any, None]]:
        """
        Fit preprocessor on training data and return processed DataFrame.

        Args:
            data:
                Raw multi-entity data loaded by DataReader. It can be:
                  - a single pandas.DataFrame
                  - or a tuple/list of DataFrames:
                    (target_df, past_cov_df, future_cov_df, static_df)

        Returns:
            Tuple compatible with the original interface:
                ({"data": df_full}, {}, None)
        """
        merged = self._merge_raw_inputs(data)
        raw_df_full = self._build_grouped_full(merged)
        df_full = raw_df_full.copy()

        # Encode categorical and scale numeric static covariates
        if self.static_cols:
            df_full = self._fit_static_encoding(df_full)

        if self.normalize:
            df_full = self._fit_and_scale(df_full)
        else:
            self.global_scaler = None
            self.entity_scalers = {}

        self.raw_df_full = raw_df_full
        self.df_full = df_full
        self.entity_dfs = self._split_by_entity(df_full)

        return {"data": df_full}, {}, None

    def predict(
        self,
        data: Union[pd.DataFrame, Tuple[pd.DataFrame, ...], List[pd.DataFrame]],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Union[Any, None]]:
        """
        Apply preprocessing for inference.

        - Does NOT fit scalers.
        - If normalize=True: use scalers fitted in fit() to transform new data.

        Args:
            data:
                Raw multi-entity data, same structure as in fit().

        Returns:
            ({"data": df_full_processed}, {}, None)
        """
        merged = self._merge_raw_inputs(data)
        raw_df_full = self._build_grouped_full(merged)
        df_full = self.transform_frame(raw_df_full)

        self.raw_df_full = raw_df_full
        self.df_full = df_full
        self.entity_dfs = self._split_by_entity(df_full)

        return {"data": df_full}, {}, None

    def _merge_raw_inputs(
        self,
        data: Union[pd.DataFrame, Tuple[pd.DataFrame, ...], List[pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        Merge the raw reader output into a single DataFrame.

        The static_grouped pipeline reader typically returns a tuple:
            (target_df, past_cov_df, future_cov_df, static_df)

        This method:
          - accepts either a single DataFrame or a tuple/list of DataFrames
          - concatenates parts along columns when needed
          - ensures time and group columns exist (single-group fallback if missing)
        """
        # Case 1: already a full DataFrame
        if isinstance(data, pd.DataFrame):
            return self._ensure_time_and_group_columns(data)

        # Case 2: tuple/list of DataFrames
        if isinstance(data, (tuple, list)):
            frames = self._collect_valid_frames(data)
            df_full = pd.concat(frames, axis=1)
            return self._ensure_time_and_group_columns(df_full)

        # Anything else is invalid
        raise ValueError(
            "StaticGroupedPreprocess expects either a pandas DataFrame "
            "or a tuple/list of DataFrames as input."
        )

    def _collect_valid_frames(
        self,
        data_seq: Union[Tuple[pd.DataFrame, ...], List[pd.DataFrame]],
    ) -> List[pd.DataFrame]:
        """
        Collect all non-None DataFrames from a tuple/list and validate types.
        """
        if not data_seq:
            raise ValueError(
                "StaticGroupedPreprocess received an empty tuple/list as input."
            )

        frames: List[pd.DataFrame] = []
        for idx, part in enumerate(data_seq):
            if part is None:
                continue
            if not isinstance(part, pd.DataFrame):
                raise ValueError(
                    "StaticGroupedPreprocess expects pandas DataFrames inside "
                    f"the tuple/list. Got type {type(part)} at index {idx}."
                )
            frames.append(part)

        if not frames:
            raise ValueError(
                "StaticGroupedPreprocess could not find any valid DataFrame "
                "inside the input tuple/list."
            )

        return frames

    def _ensure_time_and_group_columns(self, df_full: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that time_col and group_col exist.

        - If time_col is missing:
            * if index is DatetimeIndex -> reuse index as time_col
            * else -> create synthetic integer time column
        - If group_col is missing:
            * create a single-group column (all zeros)
        """
        df = df_full.copy()

        if self.time_col and self.time_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df[self.time_col] = df.index
            else:
                df[self.time_col] = range(len(df))

        if self.group_col and self.group_col not in df.columns:
            df[self.group_col] = 0

        return df

    # ------------------------------------------------------------------
    # INTERNAL HELPERS – CORE LOGIC (giữ nguyên phần dưới tương tự code bạn)
    # ------------------------------------------------------------------
    def _build_grouped_full(self, df_full: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the merged DataFrame:
        - ensure time_col and group_col are present
        - select subset columns: time, group, target, past, future, static
        - sort by group then time
        """
        if not isinstance(df_full, pd.DataFrame):
            raise ValueError(
                "StaticGroupedPreprocess expects a pandas DataFrame as input"
            )

        if (
            self.time_col not in df_full.columns
            or self.group_col not in df_full.columns
        ):
            raise ValueError(
                f"Input DataFrame must contain both time_col='{self.time_col}' "
                f"and group_col='{self.group_col}'. "
                f"Got columns: {list(df_full.columns)}"
            )

        cols = (
            [self.time_col, self.group_col]
            + self.target_cols
            + self.past_cols
            + self.future_cols
            + self.static_cols
        )
        cols = [c for c in cols if c in df_full.columns]

        base_df = df_full[cols].copy()
        base_df[self.time_col] = self._normalize_time_series(base_df[self.time_col])
        base_df = base_df.sort_values([self.group_col, self.time_col]).reset_index(
            drop=True
        )

        return base_df

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a grouped frame using fitted encoders/scalers."""
        result = self._build_grouped_full(df)

        if self.static_cols and (self._static_label_encoders or self._static_scaler):
            result = self._transform_static(result)

        if self.normalize:
            result = self._transform_with_fitted_scalers(result)

        return result

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def inverse_transform_frame(  # noqa: C901
        self,
        df: pd.DataFrame,
        *,
        value_cols: Optional[List[str]] = None,
        series_name: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Inverse-transform grouped predictions using fitted scalers."""
        if not self.normalize:
            return df

        result = df.copy()
        if value_cols is None:
            value_cols = [c for c in self.target_cols if c in result.columns]
        if not value_cols:
            value_cols = [
                c for c in result.columns if c not in [self.group_col, self.time_col]
            ]

        added_group_col = False
        if (
            self.per_entity
            and self.group_col not in result.columns
            and series_name is not None
        ):
            result[self.group_col] = series_name
            added_group_col = True

        if self.per_entity:
            if self.group_col not in result.columns:
                logger.warning(
                    "per_entity=True but group_col '%s' not present in inverse frame.",
                    self.group_col,
                )
                return df

            pieces = []
            for gid, g in result.groupby(self.group_col, sort=True):
                out = g.copy()
                scaler = self.entity_scalers.get(gid)
                if scaler is None:
                    pieces.append(out)
                    continue
                valid_cols = [c for c in value_cols if c in out.columns]
                if not valid_cols:
                    pieces.append(out)
                    continue
                feature_cols = (
                    list(self.target_cols)
                    + list(self.past_cols)
                    + list(self.future_cols)
                )
                num_features = getattr(scaler, "n_features_in_", len(feature_cols))
                dummy = np.zeros((len(out), num_features))
                for col in valid_cols:
                    if col in feature_cols:
                        idx = feature_cols.index(col)
                        if idx < num_features:
                            dummy[:, idx] = out[col].values
                inv = scaler.inverse_transform(dummy)
                for col in valid_cols:
                    if col in feature_cols:
                        idx = feature_cols.index(col)
                        if idx < num_features:
                            out[col] = inv[:, idx]
                pieces.append(out)

            result = pd.concat(pieces, ignore_index=True)
        elif self.global_scaler is not None:
            feature_cols = (
                list(self.target_cols) + list(self.past_cols) + list(self.future_cols)
            )
            valid_cols = [c for c in value_cols if c in result.columns]
            if valid_cols:
                num_features = getattr(
                    self.global_scaler, "n_features_in_", len(feature_cols)
                )
                dummy = np.zeros((len(result), num_features))
                for col in valid_cols:
                    if col in feature_cols:
                        idx = feature_cols.index(col)
                        if idx < num_features:
                            dummy[:, idx] = result[col].values
                inv = self.global_scaler.inverse_transform(dummy)
                for col in valid_cols:
                    if col in feature_cols:
                        idx = feature_cols.index(col)
                        if idx < num_features:
                            result[col] = inv[:, idx]

        result = result.sort_values(
            [c for c in [self.group_col, self.time_col] if c in result.columns]
        ).reset_index(drop=True)

        if added_group_col and self.group_col in result.columns:
            result = result.drop(columns=[self.group_col])

        return result

    def _create_scaler(self) -> StandardScaler | MinMaxScaler:
        scaler_cls = SCALER_MAP.get(self.norm_method, StandardScaler)
        return scaler_cls()

    def _fit_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers (global or per-entity) and transform DataFrame."""
        feature_cols = (
            list(self.target_cols) + list(self.past_cols) + list(self.future_cols)
        )
        feature_cols = [c for c in feature_cols if c in df.columns]

        if not feature_cols:
            logger.warning("No numeric target/past/future columns found to scale.")
            self.global_scaler = None
            self.entity_scalers = {}
            return df

        if self.per_entity:
            self.entity_scalers = {}
            groups = []

            for gid, g in df.groupby(self.group_col):
                out = g.copy()
                valid = [c for c in feature_cols if c in out.columns]
                if not valid:
                    groups.append(out)
                    continue

                scaler = self._create_scaler()
                scaler.fit(out[valid])
                out[valid] = scaler.transform(out[valid])

                self.entity_scalers[gid] = scaler
                groups.append(out)

            df_scaled = pd.concat(groups).sort_values([self.group_col, self.time_col])
            self.global_scaler = None
            return df_scaled

        scaler = self._create_scaler()
        scaler.fit(df[feature_cols])
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])

        self.global_scaler = scaler
        self.entity_scalers = {}
        return df_scaled

    # pylint: disable=too-many-locals,too-many-branches
    def _transform_with_fitted_scalers(  # noqa: C901
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform DataFrame with scalers fitted by _fit_and_scale."""
        feature_cols = (
            list(self.target_cols) + list(self.past_cols) + list(self.future_cols)
        )
        # We need the original full feature_cols order from fit time
        # The scaler was fitted on all feature_cols present during fit().
        # We assume the scaler was fitted on the exact list of feature_cols.

        valid_cols = [c for c in feature_cols if c in df.columns]

        if not valid_cols:
            logger.warning(
                "No numeric target/past/future columns found to scale in predict()."
            )
            return df

        if self.per_entity:
            if not self.entity_scalers and self.global_scaler is None:
                raise RuntimeError(
                    "No per-entity scalers have been fitted. Call fit() first "
                    "or assign entity_scalers from a checkpoint."
                )

            groups = []

            for gid, g in df.groupby(self.group_col):
                out = g.copy()
                valid = [c for c in feature_cols if c in out.columns]
                if not valid:
                    groups.append(out)
                    continue

                scaler = self.entity_scalers.get(gid)
                if scaler is None:
                    logger.warning(
                        "No scaler found for entity %r; leaving its values unscaled.",
                        gid,
                    )
                    groups.append(out)
                    continue

                num_features = getattr(scaler, "n_features_in_", len(feature_cols))
                dummy = np.zeros((len(out), num_features))

                for col in valid:
                    # In case feature_cols changed, safe lookup
                    if col in feature_cols:
                        idx = feature_cols.index(col)
                        if idx < num_features:
                            dummy[:, idx] = out[col].values

                transformed = scaler.transform(dummy)

                for col in valid:
                    if col in feature_cols:
                        idx = feature_cols.index(col)
                        if idx < num_features:
                            out[col] = transformed[:, idx]

                groups.append(out)

            df_scaled = pd.concat(groups).sort_values([self.group_col, self.time_col])
            return df_scaled

        if self.global_scaler is None:
            raise RuntimeError(
                "Global scaler has not been fitted. Call fit() first "
                "or assign global_scaler from a checkpoint."
            )

        df_scaled = df.copy()
        scaler = self.global_scaler
        num_features = getattr(scaler, "n_features_in_", len(feature_cols))
        valid = [c for c in feature_cols if c in df_scaled.columns]

        dummy = np.zeros((len(df_scaled), num_features))
        for col in valid:
            if col in feature_cols:
                idx = feature_cols.index(col)
                if idx < num_features:
                    dummy[:, idx] = df_scaled[col].values

        transformed = scaler.transform(dummy)
        for col in valid:
            if col in feature_cols:
                idx = feature_cols.index(col)
                if idx < num_features:
                    df_scaled[col] = transformed[:, idx]

        return df_scaled

    def _split_by_entity(self, df: pd.DataFrame) -> Dict[Any, pd.DataFrame]:
        """Split DataFrame into a dict mapping entity -> entity-specific DataFrame."""
        entities: Dict[Any, pd.DataFrame] = {}
        for gid, g in df.groupby(self.group_col):
            out = g.sort_values(self.time_col).reset_index(drop=True)
            entities[gid] = out
        return entities

    # ------------------------------------------------------------------
    # Static covariate encoding / scaling
    # ------------------------------------------------------------------
    def _fit_static_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical and scale numeric static columns during fit().

        Prevents raw high-magnitude values (e.g. population_density 100-5000)
        from dominating model weights when mixed with scaled dynamic features.
        """
        static_present = [c for c in self.static_cols if c in df.columns]
        if not static_present:
            return df

        result = df.copy()

        # Step 1: Label-encode categorical columns
        for col in static_present:
            if result[col].dtype == "object" or str(result[col].dtype).startswith(
                "category"
            ):
                le = LabelEncoder()
                result[col] = le.fit_transform(
                    result[col].fillna("__UNKNOWN__").astype(str)
                )
                self._static_label_encoders[col] = le

        # Step 2: Scale all static columns globally (not per-entity, because
        # static values are constant within each entity - per-entity scaling
        # would cause division-by-zero).
        scalable = [
            c for c in static_present if pd.api.types.is_numeric_dtype(result[c])
        ]
        if scalable:
            if self.group_col and self.group_col in result.columns:
                unique_vals = result.groupby(self.group_col)[scalable].first()
            else:
                unique_vals = result[scalable].drop_duplicates()

            scaler = self._create_scaler()
            scaler.fit(unique_vals)
            result[scalable] = scaler.transform(result[scalable])
            self._static_scaler = scaler
            self._static_scaler_cols = scalable

        # Store one-row-per-entity encoded + scaled static covariates
        if self.group_col and self.group_col in result.columns:
            self._encoded_static_df = (
                result.groupby(self.group_col)[static_present].first().reset_index()
            )
        else:
            self._encoded_static_df = result[static_present].iloc[[0]].copy()

        return result

    def _transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform static columns using fitted encoders and scaler."""
        static_present = [c for c in self.static_cols if c in df.columns]
        if not static_present:
            return df

        result = df.copy()

        # Step 1: Label-encode categorical columns (only if still non-numeric)
        for col in static_present:
            if (
                col in self._static_label_encoders
                and not pd.api.types.is_numeric_dtype(result[col])
            ):
                le = self._static_label_encoders[col]
                known = set(le.classes_)
                result[col] = (
                    result[col]
                    .fillna("__UNKNOWN__")
                    .astype(str)
                    .apply(
                        lambda x, _le=le, _kn=known: (
                            _le.transform([x])[0] if x in _kn else 0
                        )
                    )
                )

        # Step 2: Scale using fitted static scaler
        if self._static_scaler is not None and self._static_scaler_cols:
            cols_present = [c for c in self._static_scaler_cols if c in result.columns]
            if len(cols_present) == len(self._static_scaler_cols):
                result[cols_present] = self._static_scaler.transform(
                    result[cols_present]
                )

        return result

    def get_encoded_static(self) -> Optional[pd.DataFrame]:
        """Return encoded + scaled static covariates (one row per entity)."""
        return self._encoded_static_df

    def get_encoded_static_for_entity(self, entity_id: Any) -> Optional[pd.DataFrame]:
        """Return encoded + scaled static covariates for a specific entity."""
        if self._encoded_static_df is None:
            return None
        if self.group_col and self.group_col in self._encoded_static_df.columns:
            row = self._encoded_static_df[
                self._encoded_static_df[self.group_col] == entity_id
            ]
            if row.empty:
                return None
            return row.drop(columns=[self.group_col]).iloc[[0]]
        return self._encoded_static_df.iloc[[0]]

    def get_entity_list(self) -> List[Any]:
        """Get list of all entity IDs."""
        return list(self.entity_dfs.keys())

    def get_entity_df(self, entity: Any) -> pd.DataFrame:
        """Get DataFrame for a specific entity."""
        if entity not in self.entity_dfs:
            raise KeyError(f"Entity {entity!r} not found in preprocessed data.")
        return self.entity_dfs[entity]

    def inverse_predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(data, pd.DataFrame):
            return data

        return self.inverse_transform_frame(
            data,
            value_cols=[c for c in data.columns if c in self.target_cols],
            series_name=kwargs.get("series_name"),
        )
