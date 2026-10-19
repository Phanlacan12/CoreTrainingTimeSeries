from tsfb.base.models.darts.constants import DARTS_CLASSIC_STAT_MODEL_NAMES

DARTS_STAT_MODELS_NO_SERIES_ARG = set(DARTS_CLASSIC_STAT_MODEL_NAMES) | {
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    "StatsForecastAutoETS",
    "StatsForecastAutoARIMA",
    # "FourTheta",
}

# predefined model_args and required_args for darts models
DEEP_MODEL_REQUIRED_ARGS = {
    "input_chunk_length": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
REGRESSION_MODEL_REQUIRED_ARGS = {
    "lags": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
STAT_MODEL_REQUIRED_ARGS = {}  # type: ignore

DEEP_MODEL_ARGS = {
    "pl_trainer_kwargs": {
        "enable_progress_bar": False,
        "accelerator": "cuda",
        "devices": 1,
    }
}
