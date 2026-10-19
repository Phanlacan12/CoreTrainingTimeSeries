import os
import shutil
import sys
from pathlib import Path

import pandas as pd

FEATURESTORE_ROOT = (
    Path(__file__).resolve().parents[4] / "x06/feature_store_v4/timeseries-featurestore"
)

# Add featurestore repo to Python path to ensure it finds timeseries_featurestore module v4
if str(FEATURESTORE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURESTORE_ROOT))
from timeseries_featurestore.pipeline import Pipeline
from timeseries_featurestore.utils.logger import get_logger

logger = get_logger({"log_level": "INFO"})


def run_feature_store_pipeline():
    config_path = "configs/epidemic_fs_config.yaml"

    logger.info(f"Loading Pipeline with config: {config_path}")
    pipeline = Pipeline(config_path)

    build_dir = "build_fs_epidemic"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # 1. Run Registry Job
    logger.info("Running Registry Job...")
    pipeline.run(job_type="registry", output_dir=build_dir)
    os.environ["FEAST_REPO_DIR"] = build_dir

    output_path = "data/offline_features.parquet"

    # 2. Run Training Job
    logger.info("Running Training Job with Transformer Pipeline...")
    training_df = pipeline.run(
        job_type="training",
        mode="export_data",
        feature_service="training_service",
        start="2023-01-01",
        end="2023-02-05",  # Over 30 days
        output_path=output_path,
    )

    logger.info("Training job finished.")

    # 3. Save as CSV to be ingested by the tsfb model pipeline later
    if os.path.exists(output_path):
        result_df = pd.read_parquet(output_path)
        csv_path = "data/offline_features.csv"
        result_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(result_df)} offline feature records to {csv_path}")
        print("--- Dataframe Sample (Training) ---")
        print(result_df.head(5))
    else:
        logger.error(f"Output not found at {output_path}")

    # 4. Materialize data to online store
    logger.info("Materializing data to online store...")
    os.system(
        f"feast -c {build_dir} materialize 2023-01-01T00:00:00 2023-02-05T00:00:00"
    )
    logger.info("Online materialization complete.")


if __name__ == "__main__":
    # Ensure current working directory is e2e_cluster_example for relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_feature_store_pipeline()
