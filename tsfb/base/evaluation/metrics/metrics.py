import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, cast

import numpy as np
import pandas as pd


class BaseMetric(ABC):
    """
    Abstract base class for all metric classes.
    Provides a template for implementing custom metrics for model evaluation.
    Attributes:
        name (str): Name of the metric.
        is_higher_better (bool): Indicates if a higher value is better for this metric.
    """

    name = "base_metric"
    is_higher_better = False

    def _check_input(
        self, df: pd.DataFrame, label_col: str, predicted_col: str
    ) -> pd.DataFrame:
        """
        Validates and cleans the input DataFrame by removing
            rows with NaN values in the specified columns.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
        Returns:
            pd.DataFrame: Cleaned DataFrame with no NaN values in the specified columns.
        Raises:
            ValueError: If the DataFrame is empty or has no valid rows after cleaning.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if df[[label_col, predicted_col]].isnull().any().any():
            df = df.dropna(subset=[label_col, predicted_col])
        if df.empty:
            raise ValueError("No valid rows after dropping NaNs.")

        # Ensure numerical types for metric computation to avoid operand type errors
        df = df.astype({label_col: float, predicted_col: float})
        return df

    @abstractmethod
    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the metric score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments for metric computation.
        Returns:
            float: Computed metric score.
        """


class BaseNormMetric(ABC):
    """
    Abstract base class for all normalized metric classes.
    Provides a template for implementing metrics that
        require normalization (e.g., using a scaler).
    Attributes:
        name (str): Name of the metric.
        is_higher_better (bool): Indicates if a higher value is better for this metric.
    """

    name = "norm_metric"
    is_higher_better = False

    def _check_input(
        self, df: pd.DataFrame, label_col: str, predicted_col: str
    ) -> pd.DataFrame:
        """
        Validates and cleans the input DataFrame by
            removing rows with NaN values in the specified columns.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
        Returns:
            pd.DataFrame: Cleaned DataFrame with no NaN values in the specified columns.
        Raises:
            ValueError: If the DataFrame is empty or has no valid rows after cleaning.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if df[[label_col, predicted_col]].isnull().any().any():
            df = df.dropna(subset=[label_col, predicted_col])
        if df.empty:
            raise ValueError("No valid rows after dropping NaNs.")

        # Ensure numerical types for metric computation to avoid operand type errors
        df = df.astype({label_col: float, predicted_col: float})
        return df

    @abstractmethod
    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, scaler
    ) -> float:
        """
        Computes the normalized metric score for the given DataFrame using a scaler.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            scaler: Scaler object used to normalize the data.
        Returns:
            float: Computed normalized metric score.
        """
        df = self._check_input(df, label_col, predicted_col)
        actual = scaler.transform(df[[label_col]])
        predicted = scaler.transform(df[[predicted_col]])
        return self._compute_metric(actual.flatten(), predicted.flatten())

    @abstractmethod
    def _compute_metric(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Computes the metric score given normalized actual and predicted values.
        Args:
            actual (np.ndarray): Normalized true values.
            predicted (np.ndarray): Normalized predicted values.
        Returns:
            float: Computed metric score.
        """


class MAE(BaseMetric):
    """
    Mean Absolute Error (MAE) metric class.
    Computes the average absolute difference between true and predicted values.
    """

    name = "MAE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the MAE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            float: Computed MAE score.
        """
        df = self._check_input(df, label_col, predicted_col)
        return float(np.mean(np.abs(df[label_col] - df[predicted_col])))


class MSE(BaseMetric):
    """
    Mean Squared Error (MSE) metric class.
    Computes the average squared difference between true and predicted values.
    """

    name = "MSE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the MSE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            float: Computed MSE score.
        """
        df = self._check_input(df, label_col, predicted_col)
        return float(np.mean((df[label_col] - df[predicted_col]) ** 2))


class RMSE(BaseMetric):
    """
    Root Mean Squared Error (RMSE) metric class.
    Computes the square root of the average squared difference
        between true and predicted values.
    """

    name = "RMSE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the RMSE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            float: Computed RMSE score.
        """
        df = self._check_input(df, label_col, predicted_col)
        return float(np.sqrt(np.mean((df[label_col] - df[predicted_col]) ** 2)))


class MAPE(BaseMetric):
    """
    Mean Absolute Percentage Error (MAPE) metric class.
    Computes the average absolute percentage difference between
        true and predicted values.
    """

    name = "MAPE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the MAPE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments.
                Supports 'epsilon' to avoid division by zero.
        Returns:
            float: Computed MAPE score.
        """
        epsilon = kwargs.get("epsilon", 1e-8)
        df = self._check_input(df, label_col, predicted_col)
        actual = df[label_col].values.astype(float)
        predicted = df[predicted_col].values.astype(float)
        mask = actual != 0
        if not mask.any():
            return float("nan")

        actual = actual[mask]
        predicted = predicted[mask]
        return float(np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100)


class SMAPE(BaseMetric):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) metric class.
    Computes the average symmetric absolute percentage
        difference between true and predicted values.
    """

    name = "SMAPE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the SMAPE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            float: Computed SMAPE score.
        """
        df = self._check_input(df, label_col, predicted_col)
        actual = df[label_col].to_numpy(dtype=float)
        predicted = df[predicted_col].to_numpy(dtype=float)
        numerator = np.abs(actual - predicted)
        denominator = np.abs(actual) + np.abs(predicted)
        return float(np.mean(2.0 * numerator / denominator) * 100)


class WAPE(BaseMetric):
    """
    Weighted Absolute Percentage Error (WAPE) metric class.
    Computes the sum of absolute errors divided
        by the sum of actual values, as a percentage.
    """

    name = "WAPE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the WAPE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            float: Computed WAPE score.
        """
        df = self._check_input(df, label_col, predicted_col)
        actual = df[label_col].to_numpy(dtype=float)
        predicted = df[predicted_col].to_numpy(dtype=float)
        absolute_error = float(np.abs(actual - predicted).sum())
        absolute_actual = float(np.abs(actual).sum())
        return absolute_error / absolute_actual * 100


class MSMAPE(BaseMetric):
    """
    Modified Symmetric Mean Absolute Percentage Error (MSMAPE) metric class.
    Computes a modified version of SMAPE to handle
        small values and avoid division by zero.
    """

    name = "MSMAPE"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        """
        Computes the MSMAPE score for the given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing true and predicted values.
            label_col (str): Name of the column with true values.
            predicted_col (str): Name of the column with predicted values.
            **kwargs: Additional keyword arguments.
                Supports 'epsilon' to avoid division by zero.
        Returns:
            float: Computed MSMAPE score.
        """
        epsilon = kwargs.get("epsilon", 0.1)
        df = self._check_input(df, label_col, predicted_col)
        actual = df[label_col].to_numpy(dtype=float)
        predicted = df[predicted_col].to_numpy(dtype=float)
        comparator = np.full_like(actual, 0.5 + epsilon)
        denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
        return float(np.mean(2 * np.abs(predicted - actual) / denom) * 100)


#         return np.mean(2 * np.abs(predicted - actual) / denom) * 100


class R2(BaseMetric):
    """
    Coefficient of Determination (R²).

    Measures how well the model explains the variance of the target variable,
    compared to a baseline model that always predicts the mean.

    Ideal value: R² ∈ (-∞, 1]. R² = 1 indicates perfect prediction.
    In time series tasks, consider using R2Naive
    for a more meaningful baseline comparison.
    """

    name = "R2"
    is_higher_better = True

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        df = self._check_input(df, label_col, predicted_col)

        y_true = df[label_col].to_numpy(dtype=float)
        y_pred = df[predicted_col].to_numpy(dtype=float)
        ss_res: float = float(np.sum((y_true - y_pred) ** 2))
        y_mean = np.mean(y_true)
        ss_tot: float = float(np.sum((y_true - y_mean) ** 2))

        if np.isclose(ss_tot, 0.0):
            return 1.0 if np.allclose(y_true, y_pred) else float("nan")

        return 1.0 - ss_res / ss_tot


class R2Naive(BaseMetric):
    """
    R² relative to a naïve baseline: ŷ_naive[t] = y_true[t-1].

    Formula:
        R²_naive = 1 - SSE_model / SSE_naive
        where:
            SSE_model = ∑(y_t - ŷ_t)²
            SSE_naive = ∑(y_t - y_{t−1})²

    This is a better baseline for time series tasks than mean-based R²,
    as it evaluates whether the model improves upon simple one-step lag prediction.
    """

    name = "R2_naive"
    is_higher_better = True

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        df = self._check_input(df, label_col, predicted_col).copy()

        df["_actual_prev"] = df[label_col].shift(1)
        df = df.dropna(subset=["_actual_prev"])

        if df.empty:
            # Insufficient data to compute R²_naive (requires at least 2 points).
            return float("nan")

        y_true = df[label_col].to_numpy(dtype=float)
        y_pred = df[predicted_col].to_numpy(dtype=float)
        y_naive = df["_actual_prev"].to_numpy(dtype=float)

        sse_model: float = float(np.sum((y_true - y_pred) ** 2))
        sse_naive: float = float(np.sum((y_true - y_naive) ** 2))

        if np.isclose(sse_naive, 0.0):
            return 1.0 if np.isclose(sse_model, 0.0) else float("nan")

        return 1.0 - sse_model / sse_naive


class Coverage(BaseMetric):
    """Marginal empirical coverage for interval forecasts."""

    name = "coverage"
    is_higher_better = True

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        lower_col = kwargs.get("lower_col", "lower")
        upper_col = kwargs.get("upper_col", "upper")
        tolerance = float(kwargs.get("tolerance", 1e-12))
        if lower_col not in df.columns or upper_col not in df.columns:
            return float("nan")
        cols = [label_col, lower_col, upper_col]
        if df[cols].isnull().any().any():
            df = df.dropna(subset=cols)
        if df.empty:
            return float("nan")

        label = df[label_col].to_numpy(dtype=float)
        lower = df[lower_col].to_numpy(dtype=float)
        upper = df[upper_col].to_numpy(dtype=float)
        return float(
            np.mean((label >= (lower - tolerance)) & (label <= (upper + tolerance)))
            * 100.0
        )


class JointCoverage(BaseMetric):
    """Empirical joint coverage over forecast windows."""

    name = "joint_coverage"
    is_higher_better = True

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        lower_col = kwargs.get("lower_col", "lower")
        upper_col = kwargs.get("upper_col", "upper")
        window_col = kwargs.get("window_col", "window_id")
        tolerance = float(kwargs.get("tolerance", 1e-12))
        needed = [label_col, lower_col, upper_col, window_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return float("nan")

        work = df.dropna(subset=needed)
        if work.empty:
            return float("nan")

        inside = (
            work[label_col].to_numpy(dtype=float)
            >= (work[lower_col].to_numpy(dtype=float) - tolerance)
        ) & (
            work[label_col].to_numpy(dtype=float)
            <= (work[upper_col].to_numpy(dtype=float) + tolerance)
        )
        joint = pd.Series(inside).groupby(work[window_col].to_numpy()).all()
        return float(joint.mean() * 100.0)


class MeanIntervalWidth(BaseMetric):
    """Mean width of prediction intervals."""

    name = "mean_interval_width"

    def compute_scores(
        self, df: pd.DataFrame, label_col: str, predicted_col: str, **kwargs
    ) -> float:
        lower_col = kwargs.get("lower_col", "lower")
        upper_col = kwargs.get("upper_col", "upper")
        if lower_col not in df.columns or upper_col not in df.columns:
            return float("nan")
        cols = [lower_col, upper_col]
        if df[cols].isnull().any().any():
            df = df.dropna(subset=cols)
        if df.empty:
            return float("nan")

        width = df[upper_col].to_numpy(dtype=float) - df[lower_col].to_numpy(
            dtype=float
        )
        return float(np.mean(width))


def get_interval_metric_names() -> List[str]:
    """Return metric names that require prediction intervals."""
    return ["coverage", "joint_coverage", "mean_interval_width"]


def get_instantiated_metric_dict() -> Dict[str, BaseMetric]:
    """
    Create a dictionary of all available metric instances.

    This function dynamically instantiates all concrete (non-abstract) subclasses
    of BaseMetric defined in this module and creates a mapping from their names
    to instances.

    Returns:
        Dict[str, BaseMetric]: A dictionary mapping metric names to their corresponding
            metric instances. Each key is the name attribute of the metric class,
            and each value is an instantiated metric object.

    Example:
        {
            'mae': MAE(),
            'mse': MSE(),
            'rmse': RMSE(),
            ...
        }
    """
    res: Dict[str, BaseMetric] = {}
    for cls in BaseMetric.__subclasses__():
        if inspect.isabstract(cls) or cls.__module__ != BaseMetric.__module__:
            continue
        instance = cls()  # type: ignore[abstract]
        res[cls.name] = cast(BaseMetric, instance)
    return res
