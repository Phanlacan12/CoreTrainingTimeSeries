import os

import pandas as pd
import pytest

from tsfb.run_pipeline import PipelineExecutor


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists("./tsfb_configs/use_case.yaml"),
    reason="Config file for full pipeline not found.",
)
def test_run_and_select_best_matches_results(tmp_path):
    config_path = "./tsfb_configs/use_case.yaml"
    results_path = tmp_path / "results_benchmark.csv"

    # Run pipeline end-to-end and select best
    pe = PipelineExecutor(config_path=config_path, results_path=str(results_path))
    best = pe.run_and_select_best(
        config_path=config_path,
        metric="rmse",
        results_path=str(results_path),
        mode="auto",
    )

    df = pd.read_csv(results_path)
    assert "approach_name" in df.columns

    metric = best["metric"]
    direction = (
        "min" if metric.lower() in ["rmse", "mse", "mae", "mape", "smape"] else "max"
    )

    if direction == "min":
        expected_row = df.loc[df[metric].idxmin()]
    else:
        expected_row = df.loc[df[metric].idxmax()]

    assert best["name"] == expected_row["approach_name"]
    assert pytest.approx(best["score"], rel=1e-9) == expected_row[metric]
