import abc
import base64
import inspect
import json
import pickle
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy.constants import (
    STRATEGY_DEFAULT_CONFIG_KEY,
    STRATEGY_OPTIONAL_CONFIGS,
    STRATEGY_REQUIRED_CONFIGS,
)
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


class ResultCollector:
    """A class to collect and manage evaluation results.

    This class provides functionality to store, retrieve,
    and manage a list of evaluation results.
    """

    def __init__(self) -> None:
        """Initialize an empty ResultCollector."""
        self.results: List[Any] = []

    def add(self, result: Any) -> None:
        """Add a new result to the collector.

        Args:
            result (Any): The result to be added to the collection
        """
        self.results.append(result)

    def collect(self) -> List[Any]:
        """Retrieve all collected results.

        Returns:
            List[Any]: A list containing all collected results
        """
        return self.results

    def reset(self) -> None:
        """Clear all collected results."""
        self.results = []

    def get_size(self) -> int:
        """Get the number of collected results.

        Returns:
            int: The number of results in the collector
        """
        return len(self.results)


class Strategy(metaclass=abc.ABCMeta):
    """Base class for evaluation strategies.

    This abstract class defines the interface for implementing
    different evaluation strategies.
    It provides common functionality for configuration management and result collection.

    Attributes:
        REQUIRED_CONFIGS (List[str]): List of required configuration parameters
        DEFAULT_CONFIG_KEY (str): Default key used for configuration fallback
    """

    REQUIRED_CONFIGS = list(STRATEGY_REQUIRED_CONFIGS)
    OPTIONAL_CONFIGS: List[str] = list(STRATEGY_OPTIONAL_CONFIGS)
    DEFAULT_CONFIG_KEY = STRATEGY_DEFAULT_CONFIG_KEY

    def __init__(self, strategy_config: Dict, evaluator: Evaluator):
        """Initialize the strategy with configuration and evaluator.

        Args:
            strategy_config (Dict): Configuration dictionary for the strategy
            evaluator (Evaluator): Evaluator instance for performing evaluations
        """
        self.strategy_config = strategy_config
        self.evaluator = evaluator
        self._check_config()

    @abc.abstractmethod
    def execute(
        self,
        data_name: str,
        approach: Any,
    ) -> Any:
        """Execute the evaluation strategy.

        Args:
            data_name (str): Name of the dataset
            approach (Any): The approach to evaluate

        Returns:
            Any: Results of the strategy execution
        """

    def get_config_str(self, required_configs_only: bool = False) -> str:
        """Get configuration as a JSON string.

        Args:
            required_configs_only (bool): If True, include only required configs

        Returns:
            str: JSON string representation of the configuration
        """
        if required_configs_only:
            cfg = {
                k: v
                for k, v in self.strategy_config.items()
                if k in self.get_required_configs()
            }
            return json.dumps(cfg)
        return json.dumps(self.strategy_config, sort_keys=True)

    def _check_config(self) -> None:
        """Validate the strategy configuration.

        Checks for missing required configs and warns about extra configs.

        Raises:
            RuntimeError: If required configurations are missing
        """
        provided = set(self.strategy_config)
        required = set(self.get_required_configs())
        optional = set(self.get_optional_configs())
        missing = required - provided
        extras = provided - required - optional
        if missing:
            raise RuntimeError(f"Missing options: {', '.join(sorted(missing))}")
        if extras:
            logger.warning("Unknown options: %s", ", ".join(sorted(extras)))

    def get_collector(self) -> ResultCollector:
        """Create and return a new ResultCollector instance.

        Returns:
            ResultCollector: A new result collector instance
        """
        return ResultCollector()

    @classmethod
    @lru_cache(maxsize=1)
    def get_required_configs(cls) -> List[str]:
        """Get list of required configuration parameters.

        Returns:
            List[str]: Sorted list of required configuration parameter names
        """
        ret: List[str] = []
        for c in inspect.getmro(cls):
            if hasattr(c, "REQUIRED_CONFIGS"):
                ret.extend(c.REQUIRED_CONFIGS)
        return sorted(set(ret))

    @classmethod
    @lru_cache(maxsize=1)
    def get_optional_configs(cls) -> List[str]:
        """Get the union of optional config keys declared in the MRO."""
        ret: List[str] = []
        for c in inspect.getmro(cls):
            if hasattr(c, "OPTIONAL_CONFIGS"):
                ret.extend(c.OPTIONAL_CONFIGS)
        return sorted(set(ret))

    @staticmethod
    @abc.abstractmethod
    def accepted_metrics() -> List[str]:
        """Return list of supported metric names.

        Returns:
            List[str]: List of supported metric names
        """

    @property
    @abc.abstractmethod
    def field_names(self) -> List[str]:
        """Get names of result DataFrame columns.

        Returns:
            List[str]: List of column names for the result DataFrame
        """

    @cached_property
    def _field_name_to_idx(self) -> Dict[str, int]:
        """Create mapping from field names to their indices.

        Returns:
            Dict[str, int]: Mapping of field names to their position indices
        """
        return {name: i for i, name in enumerate(self.field_names)}

    def get_default_result(self, **kwargs) -> List[Any]:
        """Get default result list with optional overrides.

        Args:
            **kwargs: Field values to override in the default result

        Returns:
            List[Any]: Default result list with specified overrides

        Raises:
            ValueError: If an unknown field name is provided
        """
        base = self.evaluator.default_result()
        pad = [np.nan] * (len(self.field_names) - len(base))
        result = base + pad
        for k, v in kwargs.items():
            if k not in self._field_name_to_idx:
                raise ValueError(f"Unknown field name {k}")
            result[self._field_name_to_idx[k]] = v
        return result

    def _encode_data(self, data: Any) -> str:
        """Encode data as base64 string.

        Args:
            data (Any): Data to encode

        Returns:
            str: Base64 encoded string of pickled data
        """
        pickled = pickle.dumps(data)
        return base64.b64encode(pickled).decode("utf-8")

    def _get_scalar_config_value(
        self, config_name: str, data_name: Optional[str]
    ) -> Any:
        """Retrieve configuration value for given config name and data name.

        Args:
            config_name (str): Name of the configuration parameter
            data_name (Optional[str]): Name of the dataset

        Returns:
            Any: Configuration value

        Raises:
            ValueError: If config is missing or invalid
        """
        if config_name not in self.strategy_config:
            raise ValueError(f"Missing config {config_name}.")
        config_value = self.strategy_config[config_name]
        # data_module_config is a full loader config (not per-series)
        if config_name == "data_module_config":
            return config_value
        # if dict, treat as mapping from data_name to value
        if isinstance(config_value, dict):
            if (
                data_name not in config_value
                and self.DEFAULT_CONFIG_KEY not in config_value
            ):
                raise ValueError(
                    f"Config {config_name} for {data_name} missing;"
                    f" add '{data_name}' or '{self.DEFAULT_CONFIG_KEY}'."
                )
            return config_value.get(data_name, config_value[self.DEFAULT_CONFIG_KEY])
        return config_value

    def _get_meta_info(
        self, meta_info: Optional[pd.Series], field: str, default: Any
    ) -> Any:
        """Get metadata information from a Series.

        Args:
            meta_info (Optional[pd.Series]): Series containing metadata
            field (str): Field name to retrieve
            default (Any): Default value if meta_info is None

        Returns:
            Any: Value of the requested field or default

        Raises:
            KeyError: If field is missing from meta_info
        """
        if meta_info is None:
            return default
        if field not in meta_info:
            raise KeyError(f"Meta-info missing field {field}")
        return meta_info[field].item()
