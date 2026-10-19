"""
Distribution output registry — maps distribution names to parameter definitions.

Mirrors GluonTS DistributionOutput: each class knows the parameter names for a
given distribution family and how to instantiate a DistributionForecast from them.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsfb.base.forecasting.distribution_forecast import DistributionForecast


class DistributionOutput(abc.ABC):
    """
    Abstract factory: maps raw parameter arrays → a DistributionForecast.

    Subclasses declare which distribution family they represent and what
    parameter names the model must produce (e.g. 'loc', 'scale').
    """

    @property
    @abc.abstractmethod
    def distribution_name(self) -> str:
        """Unique string key, e.g. 'normal', 'student_t'."""

    @property
    @abc.abstractmethod
    def param_names(self) -> list:
        """Ordered list of parameter names the model outputs."""

    def distribution(self, params: dict) -> "DistributionForecast":
        """
        Instantiate a DistributionForecast from raw parameter arrays.

        Args:
            params: {param_name: np.ndarray of shape (prediction_length,)}

        Returns:
            DistributionForecast wrapping a scipy distribution.
        """
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from tsfb.base.forecasting.distribution_forecast import DistributionForecast

        return DistributionForecast(self.distribution_name, params)


class NormalOutput(DistributionOutput):
    """
    Normal(loc, scale) — standard Gaussian for continuous time series.

    Parameters:
        loc   — mean of the distribution
        scale — standard deviation (must be positive)
    """

    @property
    def distribution_name(self) -> str:
        return "normal"

    @property
    def param_names(self) -> list:
        return ["loc", "scale"]


class StudentTOutput(DistributionOutput):
    """
    StudentT(df, loc, scale) — heavy-tailed, robust to outliers.

    Parameters:
        df    — degrees of freedom (must be positive)
        loc   — location parameter
        scale — scale parameter (must be positive)
    """

    @property
    def distribution_name(self) -> str:
        return "student_t"

    @property
    def param_names(self) -> list:
        return ["df", "loc", "scale"]


class PoissonOutput(DistributionOutput):
    """
    Poisson(mu) — for non-negative integer count data with Var[Y] = mu.

    Parameters:
        mu — mean/rate of the distribution (must be positive)
    """

    @property
    def distribution_name(self) -> str:
        return "poisson"

    @property
    def param_names(self) -> list:
        return ["mu"]


class NegativeBinomialOutput(DistributionOutput):
    """
    NegativeBinomial(mu, alpha) — for non-negative count/demand data.

    Parameters:
        mu    — mean of the distribution (must be positive)
        alpha — dispersion parameter (must be positive)
    """

    @property
    def distribution_name(self) -> str:
        return "negative_binomial"

    @property
    def param_names(self) -> list:
        return ["mu", "alpha"]


# Registry: string key → DistributionOutput instance
DISTRIBUTION_REGISTRY: dict = {
    "normal": NormalOutput(),
    "student_t": StudentTOutput(),
    "poisson": PoissonOutput(),
    "negative_binomial": NegativeBinomialOutput(),
}


def get_distribution_output(name: str) -> DistributionOutput:
    """Look up a DistributionOutput by name; raises ValueError if unknown."""
    if name not in DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown distribution '{name}'. "
            f"Available: {list(DISTRIBUTION_REGISTRY)}"
        )
    return DISTRIBUTION_REGISTRY[name]
