# pylint: disable=duplicate-code
import os
import time
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from tsfb.base.utils.design_pattern import SingletonMeta
from tsfb.base.utils.logging import get_logger

logger = get_logger("tsfb")


class MLflowMaster(metaclass=SingletonMeta):
    """A singleton class to manage MLflow operations.
    This class initializes MLflow with a specified experiment name.
    It ensures that only one instance of MLflowMaster exists throughout the application.

    NOTE: MLFLOW_TRACKING_URI can be set to point to your MLflow tracking server.
    MLFLOW_S3_ENDPOINT_URL and MLFLOW_DEFAULT_ARTIFACT_ROOT
    can also be set if you are using S3
    for artifact storage (set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for S3 access)

    Attributes:
        mlflow (mlflow): The MLflow instance used for tracking experiments.
    """

    def __init__(
        self,
        experiment_name,
        registered_model_name="tsfb_model",
    ):
        """
        Initialize MLflowMaster.

        Args:
            experiment_name (str): Name of the MLflow experiment.
        """
        self.mlflow = mlflow
        if registered_model_name != "tsfb_model":
            logger.info(
                "MLflowMaster.registered_model_name is deprecated and ignored: %s",
                registered_model_name,
            )
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "")

        self.mlflow.set_tracking_uri(uri=tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = client.create_experiment(
                experiment_name, artifact_location=artifact_root
            )
            logger.info(
                "Created experiment: %s (ID: %s)", experiment_name, experiment_id
            )
            if artifact_root != "":
                logger.info("Experiment artifact location: %s", artifact_root)
            else:
                logger.info("No artifact location specified, using local as default.")
        else:
            logger.info(
                "Existing experiment: %s (ID: %s)",
                experiment_name,
                experiment.experiment_id,
            )
            logger.info("Existing artifact location: %s", experiment.artifact_location)

        self.mlflow.set_experiment(experiment_name)

    @staticmethod
    def check_health(
        tracking_uri: str,
        timeout: float = 30.0,
        retry_interval: float = 5.0,
    ) -> bool:
        """
        Check if the MLflow tracking server is reachable, retrying until timeout.

        For HTTP/HTTPS URIs the ``/health`` endpoint is polled; for all other URI
        schemes (sqlite, file, …) a lightweight ``search_experiments`` call is used.

        Can be called before instantiation: ``MLflowMaster.check_health(uri)``.

        Args:
            tracking_uri: MLflow tracking URI to probe.
            timeout: Total seconds to keep retrying before giving up.
            retry_interval: Seconds to wait between consecutive attempts.

        Returns:
            True if the server responded successfully, False if every attempt failed.
        """
        is_http = tracking_uri.startswith("http://") or tracking_uri.startswith(
            "https://"
        )

        def _probe() -> bool:
            if is_http:
                import urllib.error  # pylint: disable=import-outside-toplevel
                import urllib.request  # pylint: disable=import-outside-toplevel

                health_url = tracking_uri.rstrip("/") + "/health"
                try:
                    with urllib.request.urlopen(
                        health_url, timeout=max(retry_interval, 3)
                    ):
                        return True
                except (urllib.error.URLError, OSError):
                    return False
            else:
                try:
                    MlflowClient(tracking_uri).search_experiments(max_results=1)
                    return True
                except Exception:  # pylint: disable=broad-except
                    return False

        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            attempt += 1
            if _probe():
                if attempt > 1:
                    logger.info("MLflow health check passed on attempt %d.", attempt)
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            wait = min(retry_interval, remaining)
            logger.warning(
                "MLflow health check failed (attempt %d). "
                "Retrying in %.0fs (%.0fs left)…",
                attempt,
                wait,
                remaining,
            )
            time.sleep(wait)

    def has_active_run(self) -> bool:
        """Check if there's an active MLflow run."""
        return self.mlflow.active_run() is not None

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """
        Set tags on the active MLflow run.

        Args:
            tags: Dictionary of tag names and values to set
        """
        if not self.has_active_run():
            raise RuntimeError("No active MLflow run. Cannot set tags.")

        for tag_name, tag_value in tags.items():
            self.mlflow.set_tag(tag_name, tag_value)

    def get_latest_run_by_tag(
        self, tag_name: str, tag_value: str, experiment_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Find the latest run that has a specific tag.

        Args:
            tag_name: Name of the tag to search for
            tag_value: Value of the tag to match
            experiment_name: Optional experiment name to search in.
                           If None, searches in current experiment.

        Returns:
            Run ID of the latest matching run, or None if not found
        """
        client = MlflowClient()

        # Get experiment ID
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return None
            experiment_id = experiment.experiment_id
        else:
            # Use current experiment
            active_run = self.mlflow.active_run()
            if active_run:
                exp_id = active_run.info.experiment_id
                exp_name = self.mlflow.get_experiment(exp_id).name
            else:
                exp_name = "Default"
            experiment = self.mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                return None
            experiment_id = experiment.experiment_id

        # Search for runs with the specific tag
        filter_string = f"tags.`{tag_name}` = '{tag_value}'"
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs:
            return runs[0].info.run_id
        return None

    def get_latest_run_by_tags(
        self,
        tag_keys: List[str],
        tag_values: List[str],
        experiment_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find the latest run that matches multiple tags.

        Args:
            tag_keys: List of tag keys to search for
            tag_values: List of tag values to match (in same order as tag_keys)
            experiment_name: Optional experiment name to search in.
                           If None, searches in current experiment.

        Returns:
            Run ID of the latest matching run, or None if not found

        Raises:
            ValueError: If tag_keys and tag_values have different lengths
        """
        if len(tag_keys) != len(tag_values):
            raise ValueError(
                f"tag_keys and tag_values must have the same length. "
                f"Got {len(tag_keys)} keys and {len(tag_values)} values."
            )

        if not tag_keys:
            raise ValueError("tag_keys and tag_values cannot be empty")

        client = MlflowClient()

        # Get experiment ID
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return None
            experiment_id = experiment.experiment_id
        else:
            # Use current experiment
            active_run = self.mlflow.active_run()
            if active_run:
                exp_id = active_run.info.experiment_id
                exp_name = self.mlflow.get_experiment(exp_id).name
            else:
                exp_name = "Default"
            experiment = self.mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                return None
            experiment_id = experiment.experiment_id

        # Build filter string with multiple tags (AND condition)
        filter_conditions = [
            f"tags.`{key}` = '{value}'" for key, value in zip(tag_keys, tag_values)
        ]
        filter_string = " AND ".join(filter_conditions)

        # Search for runs with all specified tags
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs:
            return runs[0].info.run_id
        return None

    def get_model_uri_by_tag(
        self,
        model_name: str,
        tag_name: str,
        tag_value: str,
        experiment_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get model URI from the latest run with a specific tag.

        Args:
            model_name: Name of the model artifact
            tag_name: Name of the tag to search for
            tag_value: Value of the tag to match
            experiment_name: Optional experiment name to search in

        Returns:
            Model URI in format 'runs:/<run_id>/<model_name>', or None if not found
        """
        run_id = self.get_latest_run_by_tag(tag_name, tag_value, experiment_name)
        if run_id:
            return f"runs:/{run_id}/{model_name}"
        return None

    def get_model_uri_by_tags(
        self,
        model_name: str,
        tag_keys: List[str],
        tag_values: List[str],
        experiment_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get model URI from the latest run matching multiple tags.

        Args:
            model_name: Name of the model artifact
            tag_keys: List of tag keys to search for
            tag_values: List of tag values to match
            experiment_name: Optional experiment name to search in

        Returns:
            Model URI in format 'runs:/<run_id>/<model_name>', or None if not found
        """
        run_id = self.get_latest_run_by_tags(tag_keys, tag_values, experiment_name)
        if run_id:
            return f"runs:/{run_id}/{model_name}"
        return None
