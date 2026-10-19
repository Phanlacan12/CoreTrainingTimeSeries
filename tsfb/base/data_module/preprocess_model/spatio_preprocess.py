from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tsfb.base.data_module.preprocess_model.base_preprocess import PreprocessData
from tsfb.base.data_module.preprocess_model.constants import SCALER_MAP
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)

NormType = Literal["zscore", "minmax"]

# (wide_df, nodes, od_tensor)
SpatioRawData = Tuple[pd.DataFrame, List[str], Optional[np.ndarray]]

# Train mode output: dicts of splits
SpatioSplits = Tuple[
    Dict[str, np.ndarray],  # x_splits: {"train": X_tr, "val": X_val, "test": X_te}
    Dict[str, np.ndarray],  # y_splits: {"train": Y_tr, "val": Y_val, "test": Y_te}
    List[str],  # nodes
    Optional[np.ndarray],  # od_tensor
]

# Inference mode output: full X, y
SpatioInference = Tuple[
    np.ndarray,  # X_full (B, N, F, T_in)
    np.ndarray,  # Y_full (B, N, F, T_out)
    List[str],  # nodes
    Optional[np.ndarray],  # od_tensor
]


class SpatioPreprocess(PreprocessData):
    """Unified spatio-temporal data preprocessor:

    - Accepts wide DataFrame (time x node_feature) + nodes + optional od_tensor
    - fit(): split train/val/test, fit scaler, normalize, create sequences (B, N, F, T)
    - predict(): normalize & create sequences on full data (no split),
                 using scaler fitted during training
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        self.num_for_predict = config.get("num_for_predict", 12)
        self.num_of_hours = config.get("num_of_hours", 3)

        # normalize can be:
        #   - "zscore" / "minmax" (string)
        #   - False / None to disable
        normalize_cfg = config.get("normalize", "zscore")
        if isinstance(normalize_cfg, str):
            self.normalize_enabled = True
            norm = normalize_cfg.lower()

            if norm not in SCALER_MAP:
                logger.warning(
                    "Unknown normalize method '%s', fallback to 'zscore'", norm
                )
                norm = "zscore"
            self.scaler_type = cast(NormType, norm)

        elif normalize_cfg:
            # truthy but not a string -> enable normalize, default to zscore
            self.normalize_enabled = True
            self.scaler_type = "zscore"
        else:
            self.normalize_enabled = False
            self.scaler_type = "zscore"

        # will be set in fit/predict
        self.data: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.scaler: Optional[StandardScaler | MinMaxScaler] = None

        self.nodes: List[str] = []
        self.od_tensor: Optional[np.ndarray] = None

        self.train_data: Optional[np.ndarray] = None
        self.val_data: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None

    def fit(self, data: SpatioRawData) -> SpatioSplits:
        """Fit preprocessor & preprocess training data.

        Args:
            data: Tuple (wide_df, nodes, od_tensor):
                - wide_df: pandas.DataFrame (time x node_feature)
                - nodes: list of node names
                - od_tensor: adjacency / OD tensor (optional)

        Returns:
            (x_splits, y_splits, nodes, od_tensor)
        """
        wide_df, nodes, od_tensor = data

        self.nodes = nodes
        self.od_tensor = od_tensor
        self.data = wide_df.astype(np.float32)

        logger.info("SpatioPreprocess.fit: input wide_df shape = %s", self.data.shape)

        self._split_data()

        if self.normalize_enabled:
            self._normalize()
        else:
            # when normalization is disabled, keep raw values
            assert self.train_df is not None
            assert self.val_df is not None
            assert self.test_df is not None
            self.train_data = self.train_df.values
            self.val_data = self.val_df.values
            self.test_data = self.test_df.values

        x_splits, y_splits = self._build_xy_splits()
        return x_splits, y_splits, self.nodes, self.od_tensor

    def predict(self, data: SpatioRawData) -> SpatioInference:
        """Apply preprocessing for inference.

        - Do not split train/val/test.
        - Only normalize (if enabled) using scaler fitted in `fit()`.
        - Create sequences (X_full, Y_full).

        Args:
            data: Tuple (wide_df, nodes, od_tensor)

        Returns:
            (X_full, Y_full, nodes, od_tensor)
        """
        wide_df, nodes, od_tensor = data

        self.nodes = nodes
        self.od_tensor = od_tensor
        self.data = wide_df.astype(np.float32)

        assert self.data is not None, "Data must be provided before inference."
        data_arr = self.data.values

        if self.normalize_enabled:
            if self.scaler is None:
                raise RuntimeError(
                    "Scaler has not been fit. Call fit() first or "
                    "assign a scaler loaded from training before calling predict()."
                )
            data_arr = self.scaler.transform(data_arr)

        num_in = self.num_of_hours * self.config.get("points_per_hour", 12)
        num_out = self.num_for_predict
        n_nodes = len(self.nodes)

        x, y = self._create_sequences(data_arr, num_in, num_out)
        if x.size == 0:
            logger.warning("No sequences created in inference mode.")
            return x, y, self.nodes, self.od_tensor

        n_feat = int(x.shape[-1] / n_nodes)
        # X: (B, T_in, N, F) -> (B, N, F, T_in)
        x = self._reshape(x, num_in, n_nodes, n_feat, (0, 2, 3, 1))
        # Y: (B, T_out, N, F) -> (B, N, F, T_out)
        y = self._reshape(y, num_out, n_nodes, n_feat, (0, 2, 3, 1))

        logger.info(
            "SpatioPreprocess.predict: X shape = %s, Y shape = %s", x.shape, y.shape
        )
        return x, y, self.nodes, self.od_tensor

    def _split_data(self) -> None:
        assert self.data is not None, "Data must be set before splitting."
        ratios = self.config.get("split_ratio", {"train": 0.6, "val": 0.2, "test": 0.2})

        n = len(self.data)
        tr_end = int(n * ratios["train"])
        val_end = int(n * (ratios["train"] + ratios["val"]))

        self.train_df = self.data.iloc[:tr_end]
        self.val_df = self.data.iloc[tr_end:val_end]
        self.test_df = self.data.iloc[val_end:]

        logger.info(
            "Split sizes → train=%d, val=%d, test=%d",
            tr_end,
            val_end - tr_end,
            n - val_end,
        )

    def _normalize(self) -> None:
        assert (
            self.train_df is not None
        ), "Train split must be created before normalize."
        assert self.val_df is not None and self.test_df is not None

        scaler_cls = SCALER_MAP[self.scaler_type]
        self.scaler = scaler_cls().fit(self.train_df.values)

        self.train_data = self.scaler.transform(self.train_df)
        self.val_data = self.scaler.transform(self.val_df)
        self.test_data = self.scaler.transform(self.test_df)

        logger.info("Normalization done (%s)", self.scaler_type)

    def _create_sequences(
        self,
        data: np.ndarray,
        num_in: int,
        num_out: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slice into supervised sequences (X, y)."""
        seqs = [
            (data[i : i + num_in], data[i + num_in : i + num_in + num_out])
            for i in range(len(data) - num_in - num_out + 1)
        ]
        if not seqs:
            feat_dim = data.shape[1] if data.ndim > 1 else 1
            return (
                np.empty((0, num_in, feat_dim), dtype=data.dtype),
                np.empty((0, num_out, feat_dim), dtype=data.dtype),
            )
        x, y = zip(*seqs)
        return np.stack(x), np.stack(y)

    def _reshape(  # pylint: disable=too-many-arguments
        self,
        x: np.ndarray,
        t: int,
        n: int,
        f: int,
        transpose: Tuple[int, ...],
    ) -> np.ndarray:
        """Helper to reshape and transpose to (B, N, F, T)."""
        return x.reshape(x.shape[0], t, n, f).transpose(transpose)

    def _build_xy_splits(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create X, y for each split and reshape to (B, N, F, T)."""
        assert self.train_data is not None
        assert self.val_data is not None
        assert self.test_data is not None

        num_in = self.num_of_hours * self.config.get("points_per_hour", 12)
        num_out = self.num_for_predict
        n_nodes = len(self.nodes)

        def prep_split(split: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            x, y = self._create_sequences(split, num_in, num_out)
            if x.size == 0:
                return x, y
            n_feat = int(x.shape[-1] / n_nodes)
            # X: (B, T_in, N, F) -> (B, N, F, T_in)
            x = self._reshape(x, num_in, n_nodes, n_feat, (0, 2, 3, 1))
            # Y: (B, T_out, N, F) -> (B, N, F, T_out)
            y = self._reshape(y, num_out, n_nodes, n_feat, (0, 2, 3, 1))
            return x, y

        train_x, train_y = prep_split(self.train_data)
        val_x, val_y = prep_split(self.val_data)
        test_x, test_y = prep_split(self.test_data)

        x_splits = {
            "train": train_x,
            "val": val_x,
            "test": test_x,
        }
        y_splits = {
            "train": train_y,
            "val": val_y,
            "test": test_y,
        }

        if train_x.size > 0:
            logger.info(
                "X shape: train=%s, val=%s, test=%s",
                train_x.shape,
                val_x.shape,
                test_x.shape,
            )
            logger.info(
                "Y shape: train=%s, val=%s, test=%s",
                train_y.shape,
                val_y.shape,
                test_y.shape,
            )
        else:
            logger.warning("Empty train split after sequence creation.")

        return x_splits, y_splits

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the inverse of the normalization scaler to array x."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fit. Call fit() first.")
        return self.scaler.inverse_transform(x)

    def get_adj_matrix(self) -> Optional[np.ndarray]:
        """Return the origin-destination / adjacency tensor if available."""
        return self.od_tensor

    # pylint: disable=unused-argument
    def inverse_predict(
        self,
        data: Any,
        series_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Inverse transform sequence predictions back to original scale."""
        return data
