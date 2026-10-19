from __future__ import annotations

DARTS_CLASSIC_STAT_MODEL_NAMES = (
    "KalmanForecaster",
    "ARIMA",
    "VARIMA",
    "ExponentialSmoothing",
    "FFT",
    "Croston",
    "NaiveDrift",
    "NaiveMean",
    "NaiveSeasonal",
    "NaiveMovingAverage",
)

DARTS_DEEP_MODEL_NAMES = (
    "TCNModel",
    "TFTModel",
    "TransformerModel",
    "NHiTSModel",
    "TiDEModel",
    "BlockRNNModel",
    "RNNModel",
    "DLinearModel",
    "NBEATSModel",
    "NLinearModel",
)

DARTS_REGRESSION_MODEL_NAMES = (
    "RandomForest",
    "XGBModel",
    "CatBoostModel",
    "LightGBMModel",
    "LinearRegressionModel",
    "RegressionModel",
)

DARTS_STAT_MODEL_NAMES = (
    "AutoARIMA",
    "AutoCES",
    "AutoTheta",
    "AutoETS",
    "AutoTBATS",
) + DARTS_CLASSIC_STAT_MODEL_NAMES
