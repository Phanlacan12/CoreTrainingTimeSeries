import os
import sys
from pathlib import Path

import yaml

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfb.base.utils.parallel import ParallelBackend
from tsfb.tuning.base_tuning import BaseTuning
from tsfb.tuning.schema import TuningConfig

CONFIG_PATH = (
    REPO_ROOT / "example_for_ibm_watson_deploy/uc5_tuning/tuning_e2e_config.yaml"
)


def run_e2e_pipeline_tuning():
    print("\n--- Running E2E Pipeline Tuning ---")

    # Provider to map model name to a sampler function
    def param_sampler_provider(model_name: str):
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        tuning_cfg_dict = cfg.get("tuning", {})
        if not tuning_cfg_dict:
            return None

        # Convert simple dict to strict Pydantic model
        tuning_config = TuningConfig(**tuning_cfg_dict)

        # Helper to create the callable
        temp_tuner = BaseTuning(study_name="temp")
        return temp_tuner._create_param_sampler(tuning_config)

    tuner = BaseTuning(
        study_name="e2e_tuning_study",
        direction="minimize",
        storage="sqlite:///e2e_tuning.db",
    )

    # Ensure backend is clean before starting
    try:
        ParallelBackend().close(force=True)
    except:
        pass

    try:
        study = tuner.tune_best_from_pipeline(
            pipeline_config_path=str(CONFIG_PATH),
            param_sampler_provider=param_sampler_provider,
            results_path="benchmark_results.csv",
            n_trials=2,
            metric_name="rmse",
        )
        print(f"E2E Best Params: {study.best_params}")
        print(f"E2E Best Value: {study.best_value}")
    except Exception as e:
        print(
            f"E2E Tuning encountered an error (likely due to pipeline execution returning empty results): {e}"
        )
        # This is expected if the underlying pipeline fails to find a valid model for the dataset
        pass

    finally:
        try:
            ParallelBackend().close(force=True)
        except:
            pass


if __name__ == "__main__":
    os.makedirs("example_output", exist_ok=True)
    run_e2e_pipeline_tuning()
