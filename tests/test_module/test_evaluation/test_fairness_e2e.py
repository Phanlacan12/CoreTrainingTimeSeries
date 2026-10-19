import os
import shutil
import tempfile

import pandas as pd
import pytest
import yaml

from tsfb.base.evaluation.strategy.constants import FieldNames
from tsfb.run_pipeline import PipelineExecutor


@pytest.fixture
def temp_config_env():
    # Setup temporary directory and config
    tmp_dir = tempfile.mkdtemp()
    config_path = os.path.join(tmp_dir, "test_fairness_config.yaml")

    # Load base config content (mocking it for simplicity to avoid huge file)
    # or better: read use_case.yaml and modify it.
    base_config_path = "tsfb_configs/use_case.yaml"
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify for fast test
    config["evaluation"]["strategy_args"]["num_rollings"] = 2  # Reduce rolling windows
    config["evaluation"]["strategy_args"]["return_metrics_per_series"] = True
    config["model"]["single_model"]["model_hyper_params"]["n_epochs"] = 1

    # Disable hybrid to save time? Or keep it to test multi-approach
    # Keep it, but maybe reduce epochs on hybrid too
    config["model"]["test_hybrid"]["group"][0]["model_hyper_params"]["n_epochs"] = 1
    config["model"]["test_hybrid"]["group"][1]["model_hyper_params"]["n_epochs"] = 1

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    yield config_path, tmp_dir

    # Teardown
    shutil.rmtree(tmp_dir)


def test_fairness_e2e_pipeline(temp_config_env):
    """
    Test E2E pipeline with fairness metrics enabled.
    """
    config_path, tmp_dir = temp_config_env

    # We need to run executor.
    # PipelineExecutor usually saves to current working dir or derived from config path?
    # Let's check run_pipeline logic:
    # it saves to os.path.basename(config_path).replace(".yaml", "_results.csv") in CWD usually.

    executor = PipelineExecutor(config_path)
    executor.run()

    # Expected output file
    output_path = executor.results_path

    assert os.path.exists(output_path), f"Output file {output_path} not found"

    # Verify Content
    df = pd.read_csv(output_path)

    # 1. Check Column
    assert (
        FieldNames.METRICS_PER_SERIES in df.columns
    ), "metrics_per_series column missing"

    # 2. Check Global Row
    global_rows = df[df[FieldNames.METRICS_PER_SERIES] == "Global"]
    assert not global_rows.empty, "No Global metric rows found"

    # 3. Check Per-Series Rows
    # We expect some series names. From use_case.yaml: HUFL, MUFL, HULL, LULL
    # Check at least one
    series_rows = df[
        df[FieldNames.METRICS_PER_SERIES].isin(["HUFL", "MUFL", "HULL", "LULL"])
    ]
    assert not series_rows.empty, "No per-series metric rows found for expected groups"

    # Cleanup output file
    if os.path.exists(output_path):
        os.remove(output_path)
