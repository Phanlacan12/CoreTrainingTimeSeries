from __future__ import absolute_import

from typing import List

STRATEGY_REQUIRED_CONFIGS = ("strategy_name",)
STRATEGY_OPTIONAL_CONFIGS: tuple[str, ...] = ()
STRATEGY_DEFAULT_CONFIG_KEY = "__default__"

FORECASTING_REQUIRED_CONFIGS = (
    "seed",
    "deterministic",
    "data_module_config",
)
FORECASTING_OPTIONAL_CONFIGS = (
    "return_metrics_per_series",
    "uncertainty",
    "payload_output_dir",
    "quantile_levels",
    "probabilistic_num_samples",
    "probabilistic_output_format",
    "distribution_type",
)

QUANTILE_LEVELS_KEY = "quantile_levels"
PROBABILISTIC_NUM_SAMPLES_KEY = "probabilistic_num_samples"
PROBABILISTIC_OUTPUT_FORMAT_KEY = "probabilistic_output_format"
DISTRIBUTION_TYPE_KEY = "distribution_type"
DEFAULT_QUANTILE_LEVELS = [0.1, 0.5, 0.9]
DEFAULT_PROBABILISTIC_NUM_SAMPLES = 100
DEFAULT_PROBABILISTIC_OUTPUT_FORMAT = "samples"  # "samples" | "distribution"
DEFAULT_DISTRIBUTION_TYPE = "normal"
FORECASTING_TEST_ARTIFACT_ROOT_ENV = "TSFB_ARTIFACT_ROOT"

FIXED_FORECAST_REQUIRED_CONFIGS = FORECASTING_REQUIRED_CONFIGS + (
    "horizon",
    "save_true_pred",
    "target_channel",
)

ROLLING_FORECAST_REQUIRED_CONFIGS = FORECASTING_REQUIRED_CONFIGS + (
    "horizon",
    "stride",
    "num_rollings",
    "save_true_pred",
    "target_channel",
)


class FieldNames:
    """
    Constants for field names used in evaluation results.

    This class defines standard field names used throughout the evaluation process
    for storing model information, timing data, and results. These constants ensure
    consistency in field naming across different parts of the evaluation pipeline.
    """

    APPROACH_NAME = "approach_name"
    FILE_NAME = "file_name"
    MODEL_PARAMS = "model_params"
    STRATEGY_ARGS = "strategy_args"
    FIT_TIME = "fit_time"
    INFERENCE_TIME = "inference_time"
    INTERVAL_PAYLOAD = "interval_payload"
    # ACTUAL_DATA = "actual_data"
    # INFERENCE_DATA = "inference_data"
    LOG_INFO = "log_info"
    METRICS_PER_SERIES = "metrics_per_series"

    @classmethod
    def all_fields(cls) -> List[str]:
        """
        Get list of all available field names.

        Returns:
            List[str]: List of all field name constants defined in this class.
        """
        return [
            cls.APPROACH_NAME,
            cls.FILE_NAME,
            cls.MODEL_PARAMS,
            cls.STRATEGY_ARGS,
            cls.FIT_TIME,
            cls.INFERENCE_TIME,
            cls.INTERVAL_PAYLOAD,
            # cls.ACTUAL_DATA,
            # cls.INFERENCE_DATA,
            cls.LOG_INFO,
            cls.METRICS_PER_SERIES,
        ]
