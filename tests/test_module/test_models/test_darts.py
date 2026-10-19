import numpy as np
import pandas as pd
import pytest

from tsfb.base.models.darts.darts_factory import get_adapter_mapping
from tsfb.base.models.darts.darts_models import DartsModelAdapter


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    horizon = 10
    total_len = 100 + horizon
    dates = pd.date_range("2023-01-01", periods=total_len, freq="D")

    df = pd.DataFrame({"y": np.random.randn(100)}, index=dates[:100])

    covariates = {
        "past_covariates": pd.DataFrame(
            {"past_exog": np.random.randn(100)}, index=dates[:100]
        ),
        "future_covariates": pd.DataFrame(
            {"future_exog": np.random.randn(total_len)}, index=dates
        ),
        "static_covariates": pd.Series({"location": 1.0}),
    }

    return df, covariates


def get_model_args(model_name: str) -> dict:
    """Return appropriate args based on model type."""

    # Regression models
    if model_name in [
        "XGBModel",
        "RandomForest",
        "CatBoostModel",
        "LightGBMModel",
        "LinearRegressionModel",
        "RegressionModel",
    ]:
        return {
            "lags": [-1, -2, -3],
            "output_chunk_length": 5,
        }

    # Deep learning models
    elif model_name in [
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
    ]:
        return {
            "input_chunk_length": 10,
            "output_chunk_length": 5,
            "n_epochs": 1,
            "pl_trainer_kwargs": {"accelerator": "cpu", "devices": 1},
        }

    # Statistical models
    else:
        return {}


@pytest.mark.parametrize("model_name", list(get_adapter_mapping().keys()))
def test_darts_model_forecast_fit_and_forecast(model_name, dummy_data):
    df, covariates = dummy_data
    adapter_map = get_adapter_mapping()

    import darts.models as darts_models
    from darts.models import __all__ as all_models

    if model_name not in all_models:
        pytest.skip(f"{model_name} not available in current Darts version")

    model_class = getattr(darts_models, model_name, None)

    if model_class is None or getattr(model_class, "__module__", "").endswith(
        ".not_imported"
    ):
        pytest.skip(f"{model_name} not available or not imported")

    # VARIMA requires multivariate input
    if model_name == "VARIMA" and df.shape[1] == 1:
        df["y2"] = np.random.randn(len(df))

    # Create model
    factory_func = adapter_map[model_name]
    factory_info = factory_func(model_class)
    model_args = get_model_args(model_name)

    if model_name == "AutoTBATS":
        model_args.setdefault("season_length", 5)

    model = factory_info["model_factory"](**model_args)

    try:
        model.forecast_fit(train_data=df, covariates=covariates)
        forecast = model.forecast(horizon=5, series=df, covariates=covariates)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 5
    except NotImplementedError:
        pytest.skip(f"{model_name} not fully implemented.")
    except Exception as e:
        pytest.fail(f"{model_name} failed: {e}")


# ---------------------------------------------------------------------------
# supports_batch_forecast property
# ---------------------------------------------------------------------------


class TestDartsSuportsBatchForecast:
    def test_dl_model_returns_true(self):
        import darts.models as dm

        adapter = DartsModelAdapter(
            model_class=dm.NLinearModel,
            model_args={"input_chunk_length": 10, "output_chunk_length": 5},
        )
        assert adapter.supports_batch_forecast is True

    def test_statistical_model_returns_false(self):
        import darts.models as dm

        adapter = DartsModelAdapter(
            model_class=dm.NaiveSeasonal,
            model_args={"K": 24},
        )
        assert adapter.supports_batch_forecast is False

    def test_regression_model_no_input_chunk_returns_false(self):
        import darts.models as dm

        adapter = DartsModelAdapter(
            model_class=dm.LinearRegressionModel,
            model_args={"lags": 5, "output_chunk_length": 5},
        )
        # lags-only regression — no input_chunk_length → False
        assert adapter.supports_batch_forecast is False


# ---------------------------------------------------------------------------
# batch_forecast end-to-end — DL model only
# ---------------------------------------------------------------------------


class TestDartsBatchForecast:
    @pytest.fixture
    def nlinear_adapter(self):
        import darts.models as dm

        adapter = DartsModelAdapter(
            model_class=dm.NLinearModel,
            model_args={
                "input_chunk_length": 20,
                "output_chunk_length": 5,
                "n_epochs": 1,
                "pl_trainer_kwargs": {"accelerator": "cpu", "devices": 1},
            },
        )
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        train = pd.DataFrame({"y": np.random.randn(60)}, index=dates)
        adapter.forecast_fit(train_data=train)
        return adapter, train

    def test_batch_shape(self, nlinear_adapter):
        from tsfb.base.evaluation.strategy.rolling_batch_maker import (
            RollingForecastEvalBatchMaker,
            RollingForecastPredictBatchMaker,
        )

        adapter, series = nlinear_adapter
        # index_list: windows ending at positions 30, 40, 50
        eval_maker = RollingForecastEvalBatchMaker(series, index_list=[30, 40, 50])
        predict_maker = RollingForecastPredictBatchMaker(eval_maker)
        result = adapter.batch_forecast(horizon=5, batch_maker=predict_maker)
        assert result.shape[0] == 3  # 3 windows
        assert result.shape[1] == 5  # horizon

    def test_batch_exhausts_maker(self, nlinear_adapter):
        from tsfb.base.evaluation.strategy.rolling_batch_maker import (
            RollingForecastEvalBatchMaker,
            RollingForecastPredictBatchMaker,
        )

        adapter, series = nlinear_adapter
        eval_maker = RollingForecastEvalBatchMaker(series, index_list=[30, 40, 50])
        predict_maker = RollingForecastPredictBatchMaker(eval_maker)
        adapter.batch_forecast(horizon=5, batch_maker=predict_maker)
        assert not predict_maker.has_more_batches()

    def test_supports_flag_true(self, nlinear_adapter):
        adapter, _ = nlinear_adapter
        assert adapter.supports_batch_forecast is True
