import os
import sys
from pathlib import Path

# Add the project root to the python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tsfb.run_pipeline import PipelineExecutor

if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    config_path = "example_for_ibm_watson_deploy/uc9_error_analysis/configs/error_analysis_config.yaml"
    results_path = "results_error_analysis_demo.csv"

    # Initialize the PipelineExecutor
    executor = PipelineExecutor(config_path=config_path, results_path=results_path)

    # Run the evaluation pipeline
    # This will automatically compute per-series metrics
    # and perform error analysis on the specified columns
    executor.run()

    print(
        "Done! Check results_error_analysis_demo.csv and results_error_analysis_disease_type.csv"
    )
