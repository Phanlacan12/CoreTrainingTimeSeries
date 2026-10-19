import os
import shutil
import tempfile

import pandas as pd
import pytest
import yaml

from tsfb.base.evaluation.strategy.constants import FieldNames
from tsfb.run_pipeline import PipelineExecutor


@pytest.fixture
def temp_error_analysis_env():
    tmp_dir = tempfile.mkdtemp()
    config_path = os.path.join(tmp_dir, "test_error_analysis_config.yaml")

    base_config_path = "example_for_ibm_watson_deploy/uc3_error_analysis/configs/error_analysis_config.yaml"
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify path to be absolute or relative to project root
    config["data_module_config"]["file_path"] = "tests/data/Cluster_1_data.csv"
    config["evaluation"]["strategy_args"]["num_rollings"] = 1

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    yield config_path, tmp_dir

    shutil.rmtree(tmp_dir)


def test_error_analysis_pipeline(temp_error_analysis_env):
    """
    Test E2E pipeline with error analysis metrics grouped by static covariate.
    """
    config_path, tmp_dir = temp_error_analysis_env

    # Output paths
    output_filename = os.path.join(tmp_dir, "results_benchmark.csv")

    executor = PipelineExecutor(config_path=config_path, results_path=output_filename)
    executor.run()

    # Verify base result file
    assert os.path.exists(output_filename), f"Output file {output_filename} not found"

    # Verify error analysis files are created
    expected_disease_file = os.path.join(
        tmp_dir, "results_error_analysis_disease_type.csv"
    )
    expected_danger_file = os.path.join(
        tmp_dir, "results_error_analysis_danger_level.csv"
    )

    assert os.path.exists(
        expected_disease_file
    ), f"Error analysis file {expected_disease_file} not found"
    assert os.path.exists(
        expected_danger_file
    ), f"Error analysis file {expected_danger_file} not found"

    # Check content of the error analysis files
    df_disease = pd.read_csv(expected_disease_file)
    assert (
        "disease_type" in df_disease.columns
    ), "disease_type column missing in aggregated results"
    assert not df_disease.empty, "Disease type error analysis file is empty"

    df_danger = pd.read_csv(expected_danger_file)
    assert (
        "danger_level" in df_danger.columns
    ), "danger_level column missing in aggregated results"
    assert not df_danger.empty, "Danger level error analysis file is empty"
