# -*- coding: utf-8 -*-
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml  # type: ignore
from optuna.trial import FrozenTrial, Trial

from tsfb.base.evaluation.evaluate_model import eval_model
from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict
from tsfb.base.evaluation.strategy.fixed_forecast import FixedForecast
from tsfb.base.mlflow.approach_handle import clean_name
from tsfb.base.models.model_loader import get_single_model
from tsfb.base.utils.logging import get_logger
from tsfb.base.utils.parallel import ParallelBackend, ParallelBackendConfig
from tsfb.run_pipeline import PipelineExecutor
from tsfb.tuning.constants import TRIAL_METRICS_USER_ATTR
from tsfb.tuning.schema import ParameterType, TuningConfig

logger = get_logger(__name__)

# Disable all MLflow automatic logging integrations
mlflow.autolog(disable=True)


class TuningApproach:
    """
    Wrapper class to provide a unified interface for tuned models.

    It wraps the raw model instance and provides a safe, JSON-serializable
    configuration dictionary to avoid errors during evaluation logging.
    It also adapts method signatures (like forecast) to match what strategies expect.
    """

    def __init__(self, model: Any, name: str, config: Dict[str, Any]):
        self.model = model
        self.name = name
        self.config = config

    def forecast(
        self,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Any:
        """
        Generate forecasts using the adapted model.

        Args:
            horizon (int): Forecast horizon.
            lookback_data (pd.DataFrame): Historical data for looking back.
            covariates (Optional[Dict[str, pd.DataFrame]]): Covariates dictionary.
            **kwargs: Additional arguments.

        Returns:
            Any: Forecast result (typically np.ndarray or pd.DataFrame).
        """
        return self.model.forecast(
            horizon=horizon, series=lookback_data, covariates=covariates, **kwargs
        )

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Any:
        """
        Fit the model to the training data.

        Args:
            train_data (pd.DataFrame): Training time series data.
            covariates (Optional[Dict[str, pd.DataFrame]]): Covariates dictionary.
            **kwargs: Additional arguments.

        Returns:
            Any: The fitted model instance.
        """
        return self.model.forecast_fit(
            train_data=train_data, covariates=covariates, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        return getattr(self.model, name)


class BaseTuning:
    """
    Base class for hyperparameter tuning using Optuna.

    This class manages an Optuna study to optimize hyperparameters for a given
    time series forecasting model.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        study_name: str,
        direction: str = "minimize",
        storage: Optional[str] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        overwrite_study: bool = True,
    ):
        """
        Initialize the BaseTuning class.

        Args:
            study_name (str): Name of the Optuna study.
            direction (str): Optimization direction ('minimize' or 'maximize').
            storage (Optional[str]): Database URL for storage. Defaults to in-memory.
            sampler (Optional[optuna.samplers.BaseSampler]): Sampler object.
            pruner (Optional[optuna.pruners.BasePruner]): Pruner object.
            overwrite_study (bool): If True, automatically deletes
                any existing study with the same name to prevent configuration conflicts
        """
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.sampler = sampler
        self.pruner = pruner

        if overwrite_study and storage is not None:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                logger.info(
                    "Deleted existing Optuna study '%s' "
                    "from %s to start a fresh tuning run.",
                    study_name,
                    storage,
                )
            except Exception:
                pass

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    # pylint: disable=too-many-arguments
    def optimize(
        self,
        model_name: str,
        base_model_config: Dict[str, Any],
        evaluation_config: Dict[str, Any],
        series_list: List[str],
        param_sampler: Callable[[Trial], Dict[str, Any]],
        n_trials: int = 20,
        metric_name: str = "rmse",
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        aggregation_method: str = "mean",
        catch_exceptions: bool = True,
    ) -> optuna.study.Study:
        """
        Run the optimization process.

        Args:
            model_name (str): Expected model name (python path or registered name).
            base_model_config (Dict[str, Any]): Base configuration for the model.
            evaluation_config (Dict[str, Any]): Configuration for
                evaluation (strategy, metrics).
            series_list (List[str]): List of time series identifiers to evaluate on.
            param_sampler (Callable[[Trial], Dict[str, Any]]): Function that takes
                a trial and returns sampled parameters.
            n_trials (int): Number of trials to run.
            metric_name (str): Name of the metric to optimize.
            n_jobs (int): Number of parallel trials
                (keep to 1 if evaluation is already parallel).
            timeout (Optional[float]): Timeout in seconds.
            aggregation_method (str): Method to aggregate metrics
                across series ('mean', 'median').
            catch_exceptions (bool): Whether to catch
                exceptions during model evaluation and prune trial.

        Returns:
            optuna.study.Study: The completed Optuna study.
        """

        def objective(trial: Trial) -> float:
            model_instance = self._instantiate_model(
                trial,
                model_name,
                base_model_config,
                param_sampler,
                catch_exceptions,
            )

            return self._evaluate_trial(
                model_instance,
                series_list,
                evaluation_config,
                metric_name,
                aggregation_method,
                catch_exceptions,
            )

        logger.info(
            f"Starting optimization for {model_name} | "
            f"Trials: {n_trials} | Metric: {metric_name} ({self.direction})"
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(f"Optimization finished. Best value: {self.study.best_value}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study

    # pylint: disable=too-many-arguments
    def _instantiate_model(
        self,
        trial: Trial,
        model_name: str,
        base_model_config: Dict[str, Any],
        param_sampler: Callable[[Trial], Dict[str, Any]],
        catch_exceptions: bool,
    ) -> Any:
        # 1. Sample Hyperparameters
        sampled_params = param_sampler(trial)

        # 2. Merge with base configuration
        base_hyper_params = base_model_config.get("model_hyper_params", {})
        final_hyper_params = base_hyper_params.copy()
        final_hyper_params.update(sampled_params)

        # 3. Instantiate Model
        try:
            raw_model = get_single_model(model_name, final_hyper_params)

            safe_config = {
                "model_name": model_name,
                "model_hyper_params": final_hyper_params,
            }

            trial_model_name = f"{model_name}_trial_{trial.number}"
            model_instance = TuningApproach(raw_model, trial_model_name, safe_config)
            return model_instance

        except Exception as e:
            msg = f"Model instantiation failed: {e}"
            logger.error(msg)
            if catch_exceptions:
                raise optuna.exceptions.TrialPruned(msg)
            raise e

    # pylint: disable=too-many-arguments, too-many-locals
    def _evaluate_trial(
        self,
        model_instance: Any,
        series_list: List[str],
        evaluation_config: Dict[str, Any],
        metric_name: str,
        aggregation_method: str,
        catch_exceptions: bool,
    ) -> float:
        try:
            eval_result = eval_model(series_list, evaluation_config, model_instance)
            all_scores = self._collect_scores(eval_result, metric_name)

            if aggregation_method == "mean":
                score = float(np.mean(all_scores))
            elif aggregation_method == "median":
                score = float(np.median(all_scores))
            else:
                score = float(np.mean(all_scores))

            return score

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            msg = f"Evaluation process failed: {e}"
            logger.error(msg)
            if catch_exceptions:
                raise optuna.exceptions.TrialPruned(msg)
            raise e

    def _collect_scores(self, eval_result: Any, metric_name: str) -> List[float]:
        """Collect and validate scores from evaluation results."""
        all_scores = []
        errors = []
        for batch_df in eval_result.collect():
            # Case-insensitive column match
            col_map = {c.lower(): c for c in batch_df.columns}
            resolved_col = col_map.get(metric_name.lower())
            if resolved_col is None:
                logger.warning(
                    f"Metric '{metric_name}' not found in results. "
                    f"Available: {batch_df.columns.tolist()}"
                )
                continue

            if "log_info" in batch_df.columns:
                batch_errors = batch_df["log_info"].dropna()
                if not batch_errors.empty:
                    errors.extend(batch_errors.tolist())

            valid_scores = batch_df[resolved_col].dropna()
            valid_scores = valid_scores[~valid_scores.isin([np.inf, -np.inf])]
            all_scores.extend(valid_scores.tolist())

        if not all_scores:
            error_msg = f"No valid metric results returned. Errors: {errors[:5]}"
            logger.error(error_msg)
            raise optuna.exceptions.TrialPruned(error_msg)

        return all_scores

    def tune_from_config(
        self,
        model_entry: Dict[str, Any],
        evaluation_config: Dict[str, Any],
        series_list: List[str],
        param_sampler: Callable[[Trial], Dict[str, Any]],
        n_trials: int = 20,
        metric_name: str = "rmse",
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        aggregation_method: str = "mean",
        catch_exceptions: bool = True,
    ) -> optuna.study.Study:
        """
        Tune a model based on its configuration dictionary.

        Args:
            model_entry (Dict[str, Any]): Dictionary
                containing 'config' or full model config.
                Typically this is the output from
                `select_best_model` or a block from yaml.
            evaluation_config (Dict[str, Any]): Config for evaluation
                (strategy, metrics).
            series_list (List[str]): List of series to tune on.
            param_sampler (Callable): Function to sample params for this model.
            n_trials (int): Number of trials.
            metric_name (str): Metric to optimize.
            n_jobs (int): Number of parallel jobs.
            timeout (float): Timeout in seconds.
            aggregation_method (str): 'mean' or 'median'.
            catch_exceptions (bool): Whether to catch evaluation exceptions.

        Returns:
            optuna.study.Study: The completed study.
        """
        # Extract model name and base config
        # Handle the structure returned by
        # select_best_model which puts config in 'config'
        if "config" in model_entry:
            base_model_config = model_entry["config"]
            model_name = (
                base_model_config.get("model_name")
                or base_model_config.get("name")
                or model_entry.get("name")
            )
        else:
            # Assume model_entry IS the config
            base_model_config = model_entry
            model_name = base_model_config.get("model_name") or base_model_config.get(
                "name"
            )

        if not model_name:
            raise ValueError(
                "Could not determine model name from model_entry. "
                "Ensure 'name' or 'model_name' is present."
            )

        return self.optimize(
            model_name=model_name,
            base_model_config=base_model_config,
            evaluation_config=evaluation_config,
            series_list=series_list,
            param_sampler=param_sampler,
            n_trials=n_trials,
            metric_name=metric_name,
            n_jobs=n_jobs,
            timeout=timeout,
            aggregation_method=aggregation_method,
            catch_exceptions=catch_exceptions,
        )

    # pylint: disable=too-many-arguments
    def tune_best_from_pipeline(
        self,
        pipeline_config_path: str,
        param_sampler_provider: Callable[
            [str], Optional[Callable[[Trial], Dict[str, Any]]]
        ],
        results_path: str = "results_benchmark.csv",
        metric_name: str = "rmse",
        mode: str = "auto",
        n_trials: int = 20,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        aggregation_method: str = "mean",
        catch_exceptions: bool = True,
    ) -> optuna.study.Study:
        """
        Run the full pipeline (benchmark), select the best model, and then tune it.

        Args:
            pipeline_config_path (str): Path to use_case.yaml or pipeline config.
            param_sampler_provider (Callable[[str], Callable]): A function that takes
                a model_name (str) and returns a param_sampler function for it.
            results_path (str): Where to save benchmark results.
            metric_name (str): Metric to use for BOTH selection and tuning.
            mode (str): 'auto', 'minimize', or 'maximize' (for selection).
            n_trials (int): Tuning trials.
            n_jobs (int): Tuning jobs.
            timeout (float): Tuning timeout.
            aggregation_method (str): Aggregation for tuning score.
            catch_exceptions (bool): Catch exceptions during tuning.

        Returns:
            optuna.study.Study: The study for the best selected model.
        """

        logger.info(">>> STEP 1: Running Pipeline to determine best model...")
        executor = PipelineExecutor(
            config_path=pipeline_config_path, results_path=results_path
        )
        best_model_entry = executor.run_and_select_best(metric=metric_name, mode=mode)

        model_name = best_model_entry["name"]
        score = best_model_entry["score"]
        logger.info(
            f">>> Best Model Selection Complete. \n"
            f"Selected: {model_name} (Score: {score:.4f})"
        )

        # Retrieve configuration for tuning context
        with open(pipeline_config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)

        evaluation_config = full_cfg.get("evaluation", {})
        series_list = evaluation_config.get("series_list", ["__default__"])

        logger.info(f">>> STEP 2: Hyper-tuning {model_name}...")

        backend_cfg = full_cfg.get("backend", {})

        backend_type = backend_cfg.get("type", "ray")
        default_timeout = backend_cfg.get("default_timeout", 0.0)

        parallel_config = ParallelBackendConfig(
            backend=str(backend_type),
            n_workers=int(backend_cfg.get("n_workers", 1)),
            n_cpus=int(backend_cfg.get("n_cpus", 1)),
            gpu_devices=backend_cfg.get("gpu_devices"),
            max_tasks_per_child=backend_cfg.get("max_tasks_per_child"),
            worker_initializers=backend_cfg.get("worker_initializers"),
            default_timeout=float(default_timeout),
        )

        try:
            ParallelBackend().init(parallel_config)
        except Exception as e:
            logger.warning(f"Could not reinitialize parallel backend: {e}")

        # Get the sampler for this model
        param_sampler = param_sampler_provider(model_name)
        if not param_sampler:
            raise ValueError(f"No param_sampler provided for model: {model_name}")

        return self.tune_from_config(
            model_entry=best_model_entry,
            evaluation_config=evaluation_config,
            series_list=series_list,
            param_sampler=param_sampler,
            n_trials=n_trials,
            metric_name=metric_name,
            n_jobs=n_jobs,
            timeout=timeout,
            aggregation_method=aggregation_method,
            catch_exceptions=catch_exceptions,
        )

    def _create_param_sampler(
        self, tuning_config: TuningConfig
    ) -> Callable[[Trial], Dict[str, Any]]:
        """Create a parameter sampler function from the tuning configuration."""

        def param_sampler(trial: Trial) -> Dict[str, Any]:
            params: Dict[str, Any] = {}
            for param in tuning_config.parameters:
                if param.type == ParameterType.INT:
                    assert param.min is not None and param.max is not None
                    step_int = int(param.step) if param.step else 1
                    params[param.name] = trial.suggest_int(
                        param.name,
                        int(param.min),
                        int(param.max),
                        step=step_int,
                        log=param.log,
                    )
                elif param.type == ParameterType.FLOAT:
                    assert param.min is not None and param.max is not None
                    step_float = float(param.step) if param.step else None
                    params[param.name] = trial.suggest_float(
                        param.name,
                        float(param.min),
                        float(param.max),
                        step=step_float,
                        log=param.log,
                    )
                elif param.type == ParameterType.CATEGORICAL:
                    assert param.choices is not None
                    params[param.name] = trial.suggest_categorical(
                        param.name, param.choices
                    )
            return params

        return param_sampler

    def _format_mlflow_params(
        self, params: Dict[str, Any], base_model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format parameters for MLflow logging, such as grouping seasonal params."""
        params_to_log = params.copy()
        if all(k in params_to_log for k in ("P_seasonal", "D_seasonal", "Q_seasonal")):
            seasonal_p = params_to_log.pop("P_seasonal")
            seasonal_d = params_to_log.pop("D_seasonal")
            seasonal_q = params_to_log.pop("Q_seasonal")
            seasonal_periods = params_to_log.pop(
                "seasonal_periods",
                base_model_config.get("model_hyper_params", {}).get(
                    "seasonal_periods", 12
                ),
            )
            params_to_log["seasonal_order"] = str(
                (seasonal_p, seasonal_d, seasonal_q, seasonal_periods)
            )
        return params_to_log

    def _normalize_metric_names(self, metric_names: List[str]) -> List[str]:
        """Resolve metric names to canonical evaluator names."""
        available_metrics = get_instantiated_metric_dict()
        metric_lookup = {name.lower(): name for name in available_metrics}
        normalized: List[str] = []
        seen = set()

        for metric_name in metric_names:
            resolved_name = metric_lookup.get(metric_name.lower())
            if resolved_name is None:
                raise ValueError(f"Unknown metric: {metric_name}")
            resolved_key = resolved_name.lower()
            if resolved_key not in seen:
                normalized.append(resolved_name)
                seen.add(resolved_key)

        return normalized

    def _resolve_single_series_metrics(
        self, tuning_config: TuningConfig
    ) -> tuple[str, List[str]]:
        """Resolve the objective metric and
        eval metric list for single-series tuning."""
        metrics_eval = tuning_config.metrics_eval or [tuning_config.metric]
        normalized_metrics = self._normalize_metric_names(metrics_eval)
        objective_metric = self._normalize_metric_names([tuning_config.metric])[0]

        if objective_metric.lower() not in {
            metric_name.lower() for metric_name in normalized_metrics
        }:
            normalized_metrics.append(objective_metric)

        return objective_metric, normalized_metrics

    def _get_trial_metric_map(
        self, trial: FrozenTrial, objective_metric_name: str
    ) -> Dict[str, float]:
        """Return the metric map stored during objective evaluation."""
        metric_map = trial.user_attrs.get(TRIAL_METRICS_USER_ATTR, {})
        if isinstance(metric_map, dict):
            normalized_map = {}
            for metric_name, metric_value in metric_map.items():
                normalized_name = self._normalize_metric_names([metric_name])[0]
                normalized_map[normalized_name] = float(metric_value)
            if normalized_map:
                return normalized_map

        if trial.value is None:
            return {}
        return {objective_metric_name: float(trial.value)}

    # pylint: disable=too-many-arguments
    def _single_series_objective(
        self,
        trial: Trial,
        model_name: str,
        base_model_config: Dict[str, Any],
        param_sampler: Callable[[Trial], Dict[str, Any]],
        lookback_data: pd.DataFrame,
        y_true: np.ndarray,
        objective_metric_name: str,
        metric_names: List[str],
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        catch_exceptions: bool = True,
        log_to_mlflow: bool = True,
    ) -> float:
        """Objective function for single series tuning."""
        # 1. Instantiate Model - Moved inside try block for MLflow run context scope

        try:
            ctx: Any = mlflow.start_run(nested=True) if log_to_mlflow else nullcontext()
            with ctx:
                # 1. Instantiate Model
                model_instance = self._instantiate_model(
                    trial,
                    model_name,
                    base_model_config,
                    param_sampler,
                    catch_exceptions,
                )

                # Log parameters
                if log_to_mlflow:
                    mlflow.log_params(trial.params)

                # 2. Forecast
                model_instance.forecast_fit(
                    train_data=lookback_data, covariates=covariates
                )
                horizon = len(y_true)
                y_pred = model_instance.forecast(
                    horizon=horizon, lookback_data=lookback_data, covariates=covariates
                )

                # Ensure y_pred is numpy array
                if hasattr(y_pred, "values"):
                    y_pred = y_pred.values
                if isinstance(y_pred, list):
                    y_pred = np.array(y_pred)

                # Align y_pred shape with y_true
                if y_pred.shape != y_true.shape:
                    # If model predicted multiple columns but y_true is 1D,
                    # select only the first target column.
                    if y_pred.ndim == 2 and y_true.ndim == 1:
                        y_pred = y_pred[:, 0]
                    else:
                        y_pred = y_pred.reshape(y_true.shape)

                # 3. Evaluate
                # pylint: disable=protected-access
                metric_values = FixedForecast._evaluate_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    scaler=None,
                    metric_names=metric_names,
                )
                # Cast to Any to bypass mypy inference issue,
                global_scores: Dict[str, Any] = metric_values
                score_list = global_scores["global"]
                metric_map = {
                    metric_name: float(metric_value)
                    for metric_name, metric_value in zip(metric_names, score_list)
                }
                trial.set_user_attr(TRIAL_METRICS_USER_ATTR, metric_map)
                score = metric_map[objective_metric_name]

                if log_to_mlflow:
                    for metric_name, metric_value in metric_map.items():
                        mlflow.log_metric(metric_name, metric_value)

                return float(score)

        except Exception as e:
            msg = f"Trial failed: {e}"
            logger.error(msg)
            if catch_exceptions:
                raise optuna.exceptions.TrialPruned(msg)
            raise e

    # pylint: disable=too-many-arguments
    def tune_single_series(
        self,
        model_name: str,
        base_model_config: Dict[str, Any],
        tuning_config: TuningConfig,
        y_true: np.ndarray,
        lookback_data: pd.DataFrame,
        horizon: Optional[int] = None,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        catch_exceptions: bool = False,
        log_to_mlflow: bool = True,
        custom_tags: Optional[Dict[str, str]] = None,
    ) -> optuna.study.Study:
        """
        Tune a single model on a single series using direct forecasting and evaluation.

        Args:
            horizon: Forecasting horizon. If None, inferred from len(y_true).
        """
        objective_metric_name, metric_names = self._resolve_single_series_metrics(
            tuning_config
        )
        param_sampler = self._create_param_sampler(tuning_config)
        effective_horizon = horizon if horizon is not None else len(y_true)

        # Ensure lookback_data does not contain leftover non-numeric columns
        # (e.g. timestamp strings from grouped/static pipelines) that Darts
        # cannot convert to float.  However, columns required by the grouped
        # pipeline path (group_col, time_col, static_cols) must be kept so
        # that forecast_fit can detect and process them correctly.
        _keep_cols: set = set()
        for key in ("group_col", "group_cols", "time_col", "static_cols"):
            val = base_model_config.get(key)
            if val:
                if isinstance(val, list):
                    _keep_cols.update(val)
                else:
                    _keep_cols.add(val)

        # Safety coerce: ensure y_true is numeric (handles Object array from DB DECIMAL)
        if not np.issubdtype(y_true.dtype, np.number):
            y_true = pd.to_numeric(pd.Series(y_true), errors="coerce").values

        # Defensive coercion for lookback_data before dropping non-numeric columns.
        # This prevents dropping the target column if it was loaded as an object dtype.
        for col in lookback_data.columns:
            if not pd.api.types.is_numeric_dtype(lookback_data[col]):
                coerced = pd.to_numeric(lookback_data[col], errors="coerce")
                if coerced.notna().any():
                    lookback_data[col] = coerced

        non_numeric = lookback_data.select_dtypes(exclude="number").columns.tolist()
        cols_to_drop = [c for c in non_numeric if c not in _keep_cols]
        if cols_to_drop:
            logger.debug(
                "Dropping non-numeric columns from lookback_data: %s", cols_to_drop
            )
            lookback_data = lookback_data.drop(columns=cols_to_drop)

        self._cleanup_mlflow_runs()

        def objective(trial: Trial) -> float:
            score = self._single_series_objective(
                trial,
                model_name,
                base_model_config,
                param_sampler,
                lookback_data,
                y_true,
                objective_metric_name,
                metric_names,
                covariates,
                catch_exceptions,
                log_to_mlflow=False,  # Always False for individual trials
            )
            # Extra safety: ensure the trial didn't leave a run open
            self._cleanup_mlflow_runs()
            return score

        self.direction = tuning_config.direction
        logger.info(
            f"Starting single-series optimization for {model_name} | "
            f"Trials: {tuning_config.n_trials} | Metric: {objective_metric_name}"
        )
        self.study.optimize(
            objective,
            n_trials=tuning_config.n_trials,
            n_jobs=tuning_config.n_jobs,
            timeout=tuning_config.timeout,
            show_progress_bar=True,
        )
        logger.info(f"Optimization finished. Best value: {self.study.best_value}")
        logger.info(f"Best params: {self.study.best_params}")

        if log_to_mlflow:
            self._log_top_k_results(
                model_name,
                base_model_config,
                tuning_config,
                effective_horizon,
                lookback_data,
                covariates,
                custom_tags=custom_tags,
            )

        return self.study

    def _log_top_k_results(
        self,
        model_name: str,
        base_model_config: Dict[str, Any],
        tuning_config: TuningConfig,
        horizon: int,
        lookback_data: pd.DataFrame,
        covariates: Optional[Dict[str, pd.DataFrame]] = None,
        custom_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log top K tuning results to MLflow without re-fitting the models.

        The **best trial (rank 1)** is logged directly on the parent run
        so that its params and metrics are immediately accessible without
        drilling into nested runs.

        Trials ranked 2+ are logged as nested child runs under the same
        parent run.

        Args:
            model_name: Name of the model being tuned.
            base_model_config: Base configuration for the model.
            tuning_config: Tuning configuration (contains metric, log_top_k_trial).
            horizon: Unused here; kept for interface compatibility.
            lookback_data: Unused here; kept for interface compatibility.
            covariates: Unused here; kept for interface compatibility.
        """
        del horizon, lookback_data, covariates
        top_k = tuning_config.log_top_k_trial
        safe_model_name = clean_name(model_name)

        # Get completed trials, sorted by value according to direction
        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            logger.warning("No completed trials to log.")
            return

        reverse = self.direction == "maximize"

        def _trial_sort_value(trial: FrozenTrial) -> float:
            if trial.value is None:
                return float("-inf") if reverse else float("inf")
            return trial.value

        completed_trials.sort(key=_trial_sort_value, reverse=reverse)
        top_trials = completed_trials[:top_k]

        logger.info(
            f"Logging top {len(top_trials)} trial(s) to MLflow "
            f"(out of {len(completed_trials)} completed)..."
        )

        # Ensure a parent run exists
        parent_ctx: Any = (
            mlflow.start_run(run_name=f"tuning_{safe_model_name}")
            if mlflow.active_run() is None
            else nullcontext()
        )

        with parent_ctx:
            # Log custom tags on the parent run
            if custom_tags:
                for k, v in custom_tags.items():
                    mlflow.set_tag(k, str(v))
                logger.info("Set %d custom tag(s) on parent run.", len(custom_tags))

            for rank, trial in enumerate(top_trials):
                is_best = rank == 0

                if is_best:
                    # --- Best trial: log directly on the PARENT run ---
                    self._log_trial_to_current_run(
                        trial=trial,
                        rank=rank,
                        model_name=model_name,
                        base_model_config=base_model_config,
                        tuning_config=tuning_config,
                        is_best=True,
                        custom_tags=custom_tags,
                    )
                else:
                    # --- Non-best trials: log as nested child runs ---
                    run_name = f"rank_{rank + 1}_trial_{trial.number}"
                    try:
                        with mlflow.start_run(run_name=run_name, nested=True):
                            self._log_trial_to_current_run(
                                trial=trial,
                                rank=rank,
                                model_name=model_name,
                                base_model_config=base_model_config,
                                tuning_config=tuning_config,
                                is_best=False,
                                custom_tags=custom_tags,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log trial rank {rank + 1} "
                            f"(trial {trial.number}) to MLflow: {e}"
                        )

    def _log_trial_to_current_run(
        self,
        *,
        trial: Any,
        rank: int,
        model_name: str,
        base_model_config: Dict[str, Any],
        tuning_config: TuningConfig,
        is_best: bool,
        custom_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a single trial's params, metrics, and tags to the active MLflow run.

        Args:
            trial: The Optuna trial object.
            rank: 0-based rank of this trial.
            model_name: Original model name string.
            base_model_config: Base model configuration.
            tuning_config: Tuning configuration.
            is_best: Whether this is the best trial.
            custom_tags: Optional custom tags from user config.
        """
        final_params = base_model_config.get("model_hyper_params", {}).copy()
        final_params.update(trial.params)
        objective_metric_name, _ = self._resolve_single_series_metrics(tuning_config)
        trial_metric_map = self._get_trial_metric_map(trial, objective_metric_name)

        # Log params & metrics
        params_to_log = self._format_mlflow_params(final_params, base_model_config)
        mlflow.log_params(params_to_log)
        for metric_name, metric_value in trial_metric_map.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log tags
        mlflow.set_tag("trial_rank", str(rank + 1))
        mlflow.set_tag("is_best", str(is_best))
        mlflow.set_tag("trial_number", str(trial.number))
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("tuning_type", "single_series")
        mlflow.set_tag("direction", self.direction)
        mlflow.set_tag("model_artifact_logged", "false")

        # Log custom user tags
        if custom_tags:
            for k, v in custom_tags.items():
                mlflow.set_tag(k, str(v))

        logger.info(
            "Rank %d (trial %d): logged params/metrics to "
            "MLflow without model artifact%s",
            rank + 1,
            trial.number,
            " [BEST - on parent run]" if is_best else "",
        )

    def _cleanup_mlflow_runs(self) -> None:
        """Helper to ensure no MLflow runs are left active."""
        while mlflow.active_run():
            mlflow.end_run()
