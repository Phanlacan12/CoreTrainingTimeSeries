from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALER_MAP = {
    "zscore": StandardScaler,
    "minmax": MinMaxScaler,
}
