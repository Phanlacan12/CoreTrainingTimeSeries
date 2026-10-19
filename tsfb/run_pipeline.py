# pylint: disable=duplicate-code
from __future__ import annotations

import os
import re
import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Dict, List, Optional, cast

import mlflow
import pandas as pd

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.approaches.default import DefaultApproach
from tsfb.base.approaches.hybrid import HybridApproach
from tsfb.base.approaches.univariate import UnivariateToMultivariate
from tsfb.base.evaluation.evaluate_model import eval_model
from tsfb.base.mlflow.approach_handle import log_approach_models
from tsfb.base.mlflow.mlflow_master import MLflowMaster
from tsfb.base.models.model_loader import get_models
from tsfb.base.schema.backend_config import ParallelBackendConfig
from tsfb.base.utils.logging import get_logger, setup_logging
from tsfb.base.utils.parallel import ParallelBackend
from tsfb.base.utils.utils import load_config
from tsfb.run_and_select_best import select_best_model

warnings.filterwarnings("ignore", category=UserWarning)

TEST_ARTIFACT_ROOT_ENV = "TSFB_ARTIFACT_ROOT"
DEFAULT_RESULTS_FILENAME = "results_benchmark.csv"


def parse_and_set_env_tags(mlflow_master: MLflowMaster) -> None:
    """
    Parse tags from environment variables and set them as MLflow tags.
    Environment variables:
        tag_names: Separated list of tag names (default separator: |)
        tag_values: Separated list of tag values (default separator: |)
        tag_separator: Optional custom separator (default: |)
    """
    logger = get_logger("parse_and_set_env_tags")
    tag_names_str = os.getenv("tag_names", "")
    tag_values_str = os.getenv("tag_values", "")
    sep = os.getenv("tag_separator", "|")

    if not tag_names_str or not tag_values_str:
        return

    # Try pipe first if comma leads to mismatch, or use the explicit separator
    tag_names = [n.strip() for n in tag_names_str.split(sep) if n.strip()]
    tag_values = [v.strip() for v in tag_values_str.split(sep) if v.strip()]

    # Fallback to comma if pipe parsing yields nothing but strings were provided
    if (len(tag_names) <= 1 or len(tag_values) <= 1) and "," in tag_names_str:
        tag_names = [n.strip() for n in tag_names_str.split(",") if n.strip()]
        tag_values = [v.strip() for v in tag_values_str.split(",") if v.strip()]

    if len(tag_names) != len(tag_values):
        logger.warning(
            "tag_names (%d) and tag_values (%d) mismatch with separator '%s'. Skipping",
            len(tag_names),
            len(tag_values),
            sep,
        )
        return

    tags_dict = dict(zip(tag_names, tag_values))
    mlflow_master.set_tags(tags_dict)
    logger.info(
        "Set %d tags from environment: %s", len(tags_dict), list(tags_dict.keys())
    )


class PipelineExecutor:
    """
    Orchestrates the rolling-forecast evaluation flow from a YAML config.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    results_path : str, optional
        CSV path to save combined results. Default: "results_benchmark.csv".
    """

    def __init__(
        self, config_path: str, results_path: str = DEFAULT_RESULTS_FILENAME
    ) -> None:
        self.config_path = config_path
        self.results_path = self._resolve_results_path(results_path)

        self.cfg: dict | None = None
        self._backend_initialized: bool = False
        setup_logging(lvl="INFO", trace=False)
        self.logger = get_logger("PipelineExecutor")
        self.ml_master: Optional[MLflowMaster] = None
        self.mlflow_enabled: bool = False

    @staticmethod
    def _resolve_results_path(results_path: str) -> str:
        """Route default test results into the shared test artifact root."""
        artifact_root = os.getenv(TEST_ARTIFACT_ROOT_ENV)
        if (
            artifact_root
            and results_path == DEFAULT_RESULTS_FILENAME
            and not os.path.isabs(results_path)
        ):
            return os.path.join(artifact_root, results_path)
        return results_path

    def run(self) -> None:
        """Execute the full pipeline end-to-end."""
        try:
            self._load_config()
            self._init_backend()
            assert self.logger is not None
            self.logger.info("Building approaches…")
            approaches = self._build_approaches()

            self._run_evaluation_and_save_results(approaches)
        except Exception as exc:
            if self.logger:
                self.logger.exception("Pipeline failed: %s", exc)
            else:
                self.logger.error("[FATAL] Pipeline failed: %s", exc)
            raise
        finally:
            self._shutdown_backend()

    def _load_config(self) -> None:
        """Load YAML config into self.cfg."""
        self.logger.info("\n>>> Loading config from %s", self.config_path)
        self.cfg = load_config(self.config_path)
        if not isinstance(self.cfg, dict):
            raise ValueError("Config must be a dict loaded from YAML.")
        # Early sanity checks
        if "evaluation" not in self.cfg:
            raise KeyError("Missing 'evaluation' section in config.")
        if "strategy_args" not in self.cfg["evaluation"]:
            raise KeyError("Missing 'evaluation.strategy_args' in config.")

    def _init_backend(self) -> None:
        """Initialize the ParallelBackend from config."""
        assert self.cfg is not None, "Config must be loaded before init backend."
        self.logger.info(">>> Initializing ParallelBackend")
        backend_cfg = self.cfg.get("backend", {}) or {}

        backend_type = backend_cfg.get("type") or "local"
        backend_type = str(backend_type)

        default_timeout = backend_cfg.get("default_timeout")
        if default_timeout is None:
            default_timeout = 0.0
        else:
            default_timeout = float(default_timeout)

        parallel_config = ParallelBackendConfig(
            backend=backend_type,
            n_workers=int(backend_cfg.get("n_workers", 1)),
            n_cpus=int(backend_cfg.get("n_cpus", 1)),
            gpu_devices=backend_cfg.get("gpu_devices"),
            max_tasks_per_child=backend_cfg.get("max_tasks_per_child"),
            worker_initializers=backend_cfg.get("worker_initializers"),
            default_timeout=default_timeout,
        )
        ParallelBackend().init(parallel_config)
        self._backend_initialized = True

    def _build_approaches(self) -> Dict[str, ForecastingApproach]:
        """
        Build forecasting approaches from the 'model' section of config.

        Returns
        -------
        Dict[str, ForecastingApproach]
        """
        assert self.cfg is not None and self.logger is not None

        strategy_args = self.cfg["evaluation"]["strategy_args"]
        model_cfg = self.cfg.get("model", {}) or {}
        approaches: Dict[str, ForecastingApproach] = {}

        for name, conf in model_cfg.items():
            atype = conf.get("approach") or ("hybrid" if "group" in conf else "default")

            if atype == "default":
                factory = get_models({"models": [conf]})[0]
                app: ForecastingApproach = DefaultApproach(model_factory=factory)

            elif atype == "univariate_per_series":
                factory = get_models({"models": [conf]})[0]
                app = UnivariateToMultivariate(model_factory=factory)

            elif atype == "hybrid":
                # Merge strategy data loader config into the hybrid approach
                hybrid_full_conf = {
                    **conf,
                    "data_module_config": strategy_args["data_module_config"],
                }
                app = HybridApproach(config=hybrid_full_conf)

            else:
                raise ValueError(f"Unknown approach type '{atype}' for '{name}'")

            app.config = conf
            app.name = name
            approaches[name] = app
            self.logger.info("  • Built %s (%s)", app.__class__.__name__, name)

        if not approaches:
            raise ValueError(
                "No approaches defined. Please check 'model' section in config."
            )
        return approaches

    def _run_evaluation_and_save_results(  # noqa: C901
        self, approaches: Dict[str, ForecastingApproach]
    ) -> None:
        """Evaluate each approach, optionally log results/models to MLflow."""
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        assert self.cfg is not None
        mlflow_enabled = self._init_mlflow()
        strategy_args = dict(self.cfg["evaluation"]["strategy_args"])
        results_abs = os.path.abspath(self.results_path)
        payload_dir = os.path.join(
            os.path.dirname(results_abs) or ".",
            f"{os.path.splitext(os.path.basename(results_abs))[0]}_interval_payloads",
        )
        strategy_args.setdefault("payload_output_dir", payload_dir)
        eval_cfg = {
            "strategy_args": strategy_args,
            "metrics": self.cfg["evaluation"]["metrics"],
        }
        series_list = self.cfg["evaluation"]["series_list"]
        all_results: List[pd.DataFrame] = []

        if mlflow.active_run() is not None:
            self.logger.warning(
                "Ending previous active MLflow run before starting new one."
            )
            mlflow.end_run()

        is_single_approach = len(approaches) == 1
        experiment_name = self.cfg.get("mlflow", {}).get(
            "experiment_name", "ForecastingBenchmark"
        )
        run_name = (
            f"{experiment_name}_{list(approaches.keys())[0]}"
            if is_single_approach
            else "pipeline_run"
        )

        run_ctx: AbstractContextManager[Any] = cast(
            AbstractContextManager[Any],
            (mlflow.start_run(run_name=run_name) if mlflow_enabled else nullcontext()),
        )

        with run_ctx:
            if mlflow_enabled:
                mlflow.set_tag("pipeline", "ForecastingPipeline")
                mlflow.set_tag("config_path", self.config_path)

                # Apply tags from YAML config
                yaml_tags = self.cfg.get("mlflow", {}).get("tags", {})
                if yaml_tags and self.ml_master:
                    self.ml_master.set_tags(yaml_tags)
                    self.logger.info("Set %d tags from config.", len(yaml_tags))

                # Apply tags from environment variables
                if self.ml_master:
                    parse_and_set_env_tags(self.ml_master)

            for name, approach in approaches.items():
                self.logger.info("=== Running evaluation for %s ===", name)

                # If there's only 1 approach, don't create a nested child run
                start_child_run = mlflow_enabled and not is_single_approach

                child_ctx: AbstractContextManager[Any] = cast(
                    AbstractContextManager[Any],
                    (
                        mlflow.start_run(run_name=name, nested=True)
                        if start_child_run
                        else nullcontext()
                    ),
                )

                with child_ctx:
                    if mlflow_enabled:
                        mlflow.set_tag("approach_name", name)
                        mlflow.log_params(approach.config or {})

                    # Evaluate
                    result = eval_model(series_list, eval_cfg, approach)
                    batches = list(result.collect())
                    if not batches:
                        self.logger.warning("No results for %s", name)
                        continue

                    df = pd.concat(batches, ignore_index=True)
                    all_results.append(df)

                    # Log metrics
                    if mlflow_enabled:
                        numeric_cols = [
                            c
                            for c in df.columns
                            if pd.api.types.is_numeric_dtype(df[c])
                        ]
                        mean_metrics = df[numeric_cols].mean().to_dict()
                        for k, v in mean_metrics.items():
                            # NaN metrics break model logging on some tracking
                            # backends (sqlite UNIQUE constraint in MLflow 3.x)
                            if pd.notna(v):
                                mlflow.log_metric(k, float(v))
                        self._log_approach_models(approach, name)
                    else:
                        self.logger.info("Skipping MLflow logging for model: %s", name)

            if not all_results:
                raise RuntimeError("No evaluation results were produced.")
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_csv(self.results_path, index=False)
            self.logger.info(
                "Results saved to %s (%d rows)", self.results_path, len(final_df)
            )

            error_cols = self.cfg.get("error_analysis_column")
            if error_cols and isinstance(error_cols, list):
                try:
                    self._perform_error_analysis(final_df, error_cols)
                except Exception as eval_err:
                    self.logger.warning("Failed during error analysis: %s", eval_err)

            if mlflow_enabled:
                try:
                    mlflow.log_artifact(self.config_path)
                    mlflow.log_artifact(self.results_path)
                    mlflow.log_param("results_path", self.results_path)
                    self.logger.info("Logged results to MLflow.")
                except Exception as e:
                    self.logger.warning("Could not log artifact to MLflow: %s", e)
            else:
                self.logger.info("MLflow disabled – results saved locally only.")

    def _perform_error_analysis(
        self, final_df: pd.DataFrame, error_cols: List[str]
    ) -> None:
        """Calculate and save metric averages grouped by explicit static covariates."""
        # pylint: disable=too-many-locals,import-outside-toplevel
        from tsfb.base.evaluation.strategy.constants import FieldNames

        self.logger.info("Performing error analysis on columns: %s", error_cols)

        if not self.cfg:
            self.logger.warning("Config is None. Cannot perform error analysis.")
            return

        strategy_args = self.cfg.get("evaluation", {}).get("strategy_args", {})
        data_cfg = strategy_args.get("data_module_config")
        if not data_cfg:
            self.logger.warning(
                "No data_module_config found. Cannot perform error analysis."
            )
            return

        group_col = data_cfg.get("group_col")
        if not group_col:
            self.logger.warning(
                "Error analysis requires 'group_col' in data_module_config."
            )
            return

        try:
            from tsfb.base.data_module.factory import DataPipelineFactory

            reader, _, _ = DataPipelineFactory.create(data_cfg)
            # pylint: disable=protected-access
            df_raw = reader._dispatch_load(reader.source, {})
            # pylint: enable=protected-access
            if hasattr(df_raw, "toPandas"):
                df_raw = df_raw.toPandas()
        except Exception as e:
            self.logger.warning(
                "Failed to load raw data for error analysis mapping: %s", e
            )
            return
        per_series_df = final_df[
            (final_df[FieldNames.METRICS_PER_SERIES].notna())
            & (final_df[FieldNames.METRICS_PER_SERIES] != "Global")
        ].copy()

        if per_series_df.empty:
            self.logger.warning(
                "No per-series metrics found in final_df. "
                "Ensure return_metrics_per_series is true."
            )
            return

        metrics = self.cfg.get("evaluation", {}).get("metrics", [])
        if metrics == "all":
            from tsfb.base.evaluation.metrics.metrics import (
                get_instantiated_metric_dict,
            )

            numeric_cols = list(get_instantiated_metric_dict().keys())
        else:
            numeric_cols = metrics

        numeric_cols_matched = []
        for c in numeric_cols:
            matches = [
                col
                for col in per_series_df.columns
                if str(col).lower() == str(c).lower()
            ]
            if matches:
                numeric_cols_matched.append(matches[0])
        numeric_cols = numeric_cols_matched
        base_dir = os.path.dirname(self.results_path) or "."

        for col in error_cols:
            self._save_error_analysis_for_column(
                col, df_raw, group_col, per_series_df, numeric_cols, base_dir
            )

    def _save_error_analysis_for_column(
        self,
        col: str,
        df_raw: pd.DataFrame,
        group_col: str,
        per_series_df: pd.DataFrame,
        numeric_cols: List[str],
        base_dir: str,
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-locals,import-outside-toplevel
        # pylint: disable=redefined-outer-name,reimported
        from tsfb.base.evaluation.strategy.constants import FieldNames

        if col not in df_raw.columns:
            self.logger.warning("Error analysis column '%s' not found in dataset.", col)
            return

        if col == group_col:
            mapping_df = df_raw[[group_col]].drop_duplicates()
            mapping = {k: k for k in mapping_df[group_col]}
        else:
            mapping_df = df_raw[[group_col, col]].drop_duplicates()
            mapping = mapping_df.groupby(group_col)[col].first().to_dict()

        series_names = per_series_df[FieldNames.METRICS_PER_SERIES]
        mapped_values = series_names.map(mapping)
        if mapped_values.isna().all():
            string_mapping = {str(k): v for k, v in mapping.items()}
            mapped_values = series_names.astype(str).map(string_mapping)

        analysis_df = per_series_df.copy()
        analysis_df[col] = mapped_values

        group_cols = [
            FieldNames.APPROACH_NAME,
            FieldNames.STRATEGY_ARGS,
            FieldNames.MODEL_PARAMS,
            col,
        ]
        agg_dict = {m: "mean" for m in numeric_cols}
        agg_df = (
            analysis_df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        )

        out_file = os.path.join(base_dir, f"results_error_analysis_{col}.csv")
        agg_df.to_csv(out_file, index=False)
        self.logger.info(
            "Saved error analysis for '%s' to %s (%d rows)",
            col,
            out_file,
            len(agg_df),
        )

        if self.mlflow_enabled:
            try:
                mlflow.log_artifact(out_file)
                for _, row in agg_df.iterrows():
                    val = row[col]
                    val_str = str(val) if pd.notna(val) else "unknown"
                    for metric in numeric_cols:
                        metric_val = row[metric]
                        if pd.notna(metric_val):
                            metric_name = f"{col}_{val_str}_{metric}"
                            mlflow.log_metric(metric_name, float(metric_val))
            except Exception as e:
                self.logger.warning(
                    "Could not log error analysis artifact or metrics for %s: %s",
                    col,
                    e,
                )

    def _shutdown_backend(self) -> None:
        """Close the parallel backend if it was initialized."""
        if self._backend_initialized:
            try:
                ParallelBackend().close(force=True)
                assert self.logger is not None
                self.logger.info("Backend closed.")
            except Exception as e:
                self.logger.warning("Failed to close backend cleanly: %s", e)
        if self.logger:
            self.logger.info("Done!")

    def run_and_select_best(
        self,
        metric: str,
        config_path: Optional[str] = None,
        results_path: str = "results_benchmark.csv",
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Run the forecasting pipeline (write to results_path) and
        select the best model by metric.
        """
        effective_config = config_path or self.config_path
        if not effective_config:
            raise ValueError(
                "config_path is required (argument or instance attribute)."
            )

        effective_results = os.path.abspath(
            results_path or self.results_path or "results_benchmark.csv"
        )
        os.makedirs(os.path.dirname(effective_results) or ".", exist_ok=True)

        self.config_path = effective_config
        self.results_path = effective_results

        self.logger.info("Running pipeline with config: %s", self.config_path)
        self.run()

        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        df = pd.read_csv(self.results_path)

        if "approach_name" not in df.columns:
            cands = [c for c in df.columns if re.search(r"^approach.*name$", c, re.I)]
            if cands:
                df = df.rename(columns={cands[0]: "approach_name"})
            else:
                raise ValueError("Column 'approach_name' not found in results file.")

        return select_best_model(
            df=df, config_path=self.config_path, metric=metric, mode=mode
        )

    def _init_mlflow(self) -> bool:
        """Initialize MLflow (via MLflowMaster) if enabled in config."""
        assert self.cfg is not None, "Config must be loaded before initializing MLflow."
        mlflow_cfg = self.cfg.get("mlflow", {}) or {}
        self.mlflow_enabled = mlflow_cfg.get("enabled", False)
        experiment_name = mlflow_cfg.get("experiment_name", "ForecastingBenchmark")

        if not self.mlflow_enabled:
            self.logger.info("MLflow logging disabled.")
            self.ml_master = None
            return False

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        hc_timeout = float(mlflow_cfg.get("health_check_timeout", 30))
        hc_interval = float(mlflow_cfg.get("health_check_retry_interval", 5))

        self.logger.info(
            "MLflow enabled. Running health check on "
            "'%s' (timeout=%.0fs, interval=%.0fs)…",
            tracking_uri,
            hc_timeout,
            hc_interval,
        )
        healthy = MLflowMaster.check_health(
            tracking_uri, timeout=hc_timeout, retry_interval=hc_interval
        )
        if not healthy:
            self.logger.warning(
                "MLflow server at '%s' is not reachable after %.0fs. "
                "MLflow logging will be DISABLED – training continues without it.",
                tracking_uri,
                hc_timeout,
            )
            self.mlflow_enabled = False
            self.ml_master = None
            return False

        self.ml_master = MLflowMaster(experiment_name)
        self.logger.info("MLflow enabled. Experiment: %s", experiment_name)
        return True

    def _log_approach_models(
        self, approach: ForecastingApproach, approach_name: str
    ) -> None:
        assert self.cfg is not None
        log_approach_models(
            cfg=self.cfg,
            approach=approach,
            approach_name=approach_name,
            mlflow_enabled=self.mlflow_enabled,
            run_logger=self.logger,
        )
