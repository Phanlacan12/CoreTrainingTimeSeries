# pylint: disable=duplicate-code
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from tsfb.base.data_module.preprocess_model.base_preprocess import PreprocessData
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)

NormType = Literal["zscore"]

# (wide_df, nodes, od_tensor)
SpatioRawData = Tuple[pd.DataFrame, List[str], Optional[np.ndarray]]

# Train mode output: dicts of splits
ASTGCRNSplits = Tuple[
    Dict[str, np.ndarray],  # x_splits: {"train": X_tr, "val": X_val, "test": X_te}
    Dict[str, np.ndarray],  # y_splits: {"train": Y_tr, "val": Y_val, "test": Y_te}
    List[str],  # nodes
    Optional[np.ndarray],  # od_tensor
]

# Inference mode output: full X, y
ASTGCRNInference = Tuple[
    np.ndarray,  # X_full (B, N, F, t_in)
    np.ndarray,  # Y_full (B, N, T_out)
    List[str],  # nodes
    Optional[np.ndarray],  # od_tensor
]


class ASTGCRNPreprocess(PreprocessData):
    """
    ASTGCRN-style spatio-temporal preprocessor.

    Goal: emulate the original `ASTGCRNDataLoader` preprocessing behavior,
    while separating preprocessing into its own component.

    Behavior:
    - Accepts wide_df (time x node_feature) + nodes + optional od_tensor
    - Creates sliding windows:
        * X: (B, N, F, t_in)
        * Y: (B, N, T_out)  (targets are kept in original scale)
    - Splits sequences into train/val/test according to config["split_ratio"]
    - Applies Z-score normalization based on train X:
        X_norm = (X - mean) / std
        where mean and std are scalars computed over the train X
    - `fit(...)` is used for training:
        returns (x_splits_norm, y_splits_raw, nodes, od_tensor)
    - `predict(...)` is used for inference:
        uses mean/std fitted on train to normalize X_full
        and returns (X_full_norm, Y_full_raw, nodes, od_tensor)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        # Time configuration (ASTGCN-style)
        self.num_of_hours = config.get("num_of_hours", 1)
        self.num_of_days = config.get("num_of_days", 0)
        self.num_of_weeks = config.get("num_of_weeks", 0)
        self.num_for_predict = config.get("num_for_predict", 12)
        self.points_per_hour = config.get("points_per_hour", 12)

        # Total input steps t_in
        self.t_in = (
            self.num_of_hours * self.points_per_hour
            + self.num_of_days * 24 * self.points_per_hour
            + self.num_of_weeks * 7 * 24 * self.points_per_hour
        )

        # Original ASTGCRN supports zscore only; enable normalization by default
        self.normalize_enabled: bool = bool(config.get("normalize", True))
        self.scaler_type: NormType = "zscore"

        # Will be assigned in fit/predict
        self.data: Optional[pd.DataFrame] = None
        self.nodes: List[str] = []
        self.od_tensor: Optional[np.ndarray] = None

        # Train/val/test sequences (raw and normalized)
        self.train_x: Optional[np.ndarray] = None  # raw X train
        self.val_x: Optional[np.ndarray] = None
        self.test_x: Optional[np.ndarray] = None

        self.train_x_norm: Optional[np.ndarray] = None
        self.val_x_norm: Optional[np.ndarray] = None
        self.test_x_norm: Optional[np.ndarray] = None

        self.train_y: Optional[np.ndarray] = None  # Y kept in original scale
        self.val_y: Optional[np.ndarray] = None
        self.test_y: Optional[np.ndarray] = None

        # mean/std for inverse transform (broadcastable to X: (B, N, F, t_in))
        self._mean: Optional[np.ndarray] = None  # shape (1, 1, 1, 1)
        self._std: Optional[np.ndarray] = None  # shape (1, 1, 1, 1)

        logger.info(
            "ASTGCRNPreprocess time config → hours=%d, days=%d, weeks=%d, "
            "total_input=%d, output=%d",
            self.num_of_hours,
            self.num_of_days,
            self.num_of_weeks,
            self.t_in,
            self.num_for_predict,
        )

    def fit(self, data: SpatioRawData) -> ASTGCRNSplits:
        """
        Fit preprocessor & preprocess training data (ASTGCN-style).

        Args:
            data: Tuple (wide_df, nodes, od_tensor):
                - wide_df: pandas.DataFrame (time x node_feature)
                - nodes: list of node names
                - od_tensor: adjacency / OD tensor (optional)

        Returns:
            (x_splits_norm, y_splits_raw, nodes, od_tensor)
        """
        wide_df, nodes, od_tensor = data

        self.nodes = nodes
        self.od_tensor = od_tensor
        self.data = wide_df.astype(np.float32)

        logger.info("ASTGCRNPreprocess.fit: input wide_df shape = %s", self.data.shape)

        full = np.asarray(self.data, dtype=np.float32)

        # 1) Create sliding sequences for the entire series
        xs, ys = self._create_sequences(full)

        # 2) Split train/val/test and normalize using train X
        self._split_and_normalize(xs, ys)

        # 3) Build output dicts (normalized X, raw Y)
        x_splits: Dict[str, np.ndarray] = {
            "train": (
                self.train_x_norm
                if self.train_x_norm is not None
                else self._empty_x_like(xs)
            ),
            "val": (
                self.val_x_norm
                if self.val_x_norm is not None
                else self._empty_x_like(xs)
            ),
            "test": (
                self.test_x_norm
                if self.test_x_norm is not None
                else self._empty_x_like(xs)
            ),
        }
        y_splits: Dict[str, np.ndarray] = {
            "train": (
                self.train_y if self.train_y is not None else self._empty_y_like(ys)
            ),
            "val": self.val_y if self.val_y is not None else self._empty_y_like(ys),
            "test": self.test_y if self.test_y is not None else self._empty_y_like(ys),
        }

        if x_splits["train"].size > 0:
            logger.info(
                "ASTGCRNPreprocess: X shape train=%s, val=%s, test=%s",
                x_splits["train"].shape,
                x_splits["val"].shape,
                x_splits["test"].shape,
            )
            logger.info(
                "ASTGCRNPreprocess: Y shape train=%s, val=%s, test=%s",
                y_splits["train"].shape,
                y_splits["val"].shape,
                y_splits["test"].shape,
            )
        else:
            logger.warning(
                "ASTGCRNPreprocess: empty train split after sequence creation."
            )

        return x_splits, y_splits, self.nodes, self.od_tensor

    def predict(self, data: SpatioRawData) -> ASTGCRNInference:
        """
        Apply ASTGCRN-style preprocessing for inference.

        - Do not split into train/val/test.
        - If normalize=True: use mean/std fitted on train to z-score X.
        - Y is always kept in original scale.

        Args:
            data: Tuple (wide_df, nodes, od_tensor)

        Returns:
            (X_full_norm, Y_full_raw, nodes, od_tensor)
        """
        wide_df, nodes, od_tensor = data

        self.nodes = nodes
        self.od_tensor = od_tensor
        self.data = wide_df.astype(np.float32)

        assert self.data is not None, "Data must be provided before inference."
        full = np.asarray(self.data, dtype=np.float32)

        xs, ys = self._create_sequences(full)
        if xs.size == 0:
            logger.warning("ASTGCRNPreprocess.predict: no sequences created.")
            return xs, ys, self.nodes, self.od_tensor

        if self.normalize_enabled:
            if self._mean is None or self._std is None:
                raise RuntimeError(
                    "Normalization parameters (mean/std) have not been fitted. "
                    "Call fit() first before predict()."
                )
            xs_norm = ((xs - self._mean) / self._std).astype(np.float32)
        else:
            xs_norm = xs

        logger.info(
            "ASTGCRNPreprocess.predict: X shape = %s, Y shape = %s",
            xs_norm.shape,
            ys.shape,
        )
        return xs_norm, ys, self.nodes, self.od_tensor

    def _create_sequences(self, full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows similar to ASTGCRNDataLoader._create_sliding_sequences:

        - Input:
            full: (T, N * F)
        - Output:
            X: (S, N, F, t_in)
            Y: (S, N, T_out)  (feature dim squeezed)
        """
        n_nodes = len(self.nodes)
        if n_nodes <= 0:
            logger.warning("ASTGCRNPreprocess: number of nodes is zero.")
            # unknown feature dim -> assume 1 to avoid shape errors
            feat_dim = 1
        else:
            feat_dim = int(full.shape[1] / n_nodes) if full.ndim == 2 else 1

        total_t = full.shape[0]

        xs_list: List[np.ndarray] = []
        ys_list: List[np.ndarray] = []

        max_i = total_t - self.t_in - self.num_for_predict + 1
        for i in range(max(0, max_i)):
            x = full[i : i + self.t_in]  # (t_in, N*F)
            y = full[
                i + self.t_in : i + self.t_in + self.num_for_predict
            ]  # (T_out, N*F)
            xs_list.append(x)
            ys_list.append(y)

        if not xs_list:
            # Empty sequences -> create empty arrays with valid shapes
            empty_x = np.empty((0, n_nodes, feat_dim, self.t_in), dtype=np.float32)
            empty_y = np.empty((0, n_nodes, self.num_for_predict), dtype=np.float32)
            return empty_x, empty_y

        xs = np.stack(xs_list)  # (S, t_in, N*F)
        ys = np.stack(ys_list)  # (S, T_out, N*F)

        # X: (S, t_in, N, F) -> (S, N, F, t_in)
        xs = xs.reshape(xs.shape[0], self.t_in, n_nodes, feat_dim).transpose(0, 2, 3, 1)

        # Y: (S, T_out, N, F) -> (S, N, F, T_out) -> squeeze F dim -> (S, N, T_out)
        ys = ys.reshape(ys.shape[0], self.num_for_predict, n_nodes, feat_dim).transpose(
            0, 2, 3, 1
        )
        ys = np.squeeze(ys, axis=2)  # (S, N, T_out)

        return xs.astype(np.float32), ys.astype(np.float32)

    def _split_and_normalize(  # pylint: disable=too-many-locals
        self,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        """
        Split sequences along the S dimension (number of sequences) and z-score
        normalize according to train X (mimicking
        ASTGCRNDataLoader._split_and_normalize).

        Sets:
            - self.train_x_norm, self.val_x_norm, self.test_x_norm
            - self.train_y, self.val_y, self.test_y
            - self._mean, self._std (shape (1, 1, 1, 1))
        """
        s = xs.shape[0]
        ratios = self.config.get("split_ratio", {"train": 0.6, "val": 0.2, "test": 0.2})
        s_tr = int(s * ratios["train"])
        s_val = s_tr + int(s * ratios["val"])

        train_x, val_x, test_x = xs[:s_tr], xs[s_tr:s_val], xs[s_val:]
        train_y, val_y, test_y = ys[:s_tr], ys[s_tr:s_val], ys[s_val:]

        self.train_x = train_x
        self.val_x = val_x
        self.test_x = test_x

        self.train_y = train_y
        self.val_y = val_y
        self.test_y = test_y

        if not self.normalize_enabled or train_x.size == 0:
            # No normalization or no training data
            if train_x.size == 0:
                logger.warning("ASTGCRNPreprocess: no training sequences available.")
            # neutral mean/std (inverse_transform will be identity)
            self._mean = np.array(0.0, dtype=np.float32).reshape((1, 1, 1, 1))
            self._std = np.array(1.0, dtype=np.float32).reshape((1, 1, 1, 1))

            self.train_x_norm = train_x
            self.val_x_norm = val_x
            self.test_x_norm = test_x
        else:
            mean_value = train_x.mean()
            std_value = train_x.std()

            mean = np.array(mean_value, dtype=np.float32).reshape((1, 1, 1, 1))
            std = np.array(std_value, dtype=np.float32).reshape((1, 1, 1, 1))
            std[std == 0] = 1.0

            def normalize(x: np.ndarray) -> np.ndarray:
                return ((x - mean) / std).astype(np.float32)

            self.train_x_norm = normalize(train_x)
            self.val_x_norm = normalize(val_x)
            self.test_x_norm = normalize(test_x)

            self._mean = mean
            self._std = std

        logger.info(
            "ASTGCRNPreprocess: sequence counts → train=%d, val=%d, test=%d (sum=%d)",
            len(train_x),
            len(val_x),
            len(test_x),
            s,
        )
        assert self._mean is not None
        assert self._std is not None
        logger.info(
            "ASTGCRNPreprocess: Z-score mean=%s, std=%s",
            self._mean.flatten(),
            self._std.flatten(),
        )

    def _empty_x_like(self, xs: np.ndarray) -> np.ndarray:
        """Helper: create an empty X with same (N, F, t_in)."""
        if xs.size == 0:
            # default (0, N, F, t_in) with N,F from config/nodes
            n_nodes = len(self.nodes)
            feat_dim = (
                int(self.data.shape[1] / max(1, n_nodes))
                if self.data is not None and n_nodes > 0
                else 1
            )
            return np.empty((0, n_nodes, feat_dim, self.t_in), dtype=np.float32)
        return np.empty_like(xs[:0])

    def _empty_y_like(self, ys: np.ndarray) -> np.ndarray:
        """Helper: create an empty Y with same (N, T_out)."""
        if ys.size == 0:
            n_nodes = len(self.nodes)
            return np.empty((0, n_nodes, self.num_for_predict), dtype=np.float32)
        return np.empty_like(ys[:0])

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse z-score normalization for X.

        Args:
            x: Normalized array (broadcastable to shape (1, 1, 1, 1)).

        Returns:
            Array rescaled to original scale.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Normalization parameters (mean/std) are not available. "
                "Call fit() first."
            )
        return x * self._std + self._mean

    @property
    def normalization_stats(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return fitted normalization statistics as `(mean, std)`."""
        return self._mean, self._std

    def normalize_inputs(self, x: np.ndarray) -> np.ndarray:
        """Normalize inference inputs with the fitted z-score statistics."""
        if not self.normalize_enabled:
            return x.astype(np.float32, copy=False)
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Normalization parameters (mean/std) are not available. "
                "Call fit() first."
            )
        return ((x - self._mean) / self._std).astype(np.float32)

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
