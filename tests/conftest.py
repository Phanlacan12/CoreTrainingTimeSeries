from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _cleanup_repo_artifact_dirs(repo_root: Path) -> None:
    """Remove repo-root artifact directories that tests should not leave behind."""
    for path in repo_root.glob("*_interval_payloads"):
        shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(repo_root / "interval_payloads", ignore_errors=True)
    shutil.rmtree(repo_root / "experiments", ignore_errors=True)
    shutil.rmtree(repo_root / "mlruns", ignore_errors=True)
    # Chronos finetune artifacts written to cwd when no checkpoint_dir is set
    for path in repo_root.glob("*-finetuned"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    mlflow_db = repo_root / "mlflow.db"
    if mlflow_db.exists():
        mlflow_db.unlink(missing_ok=True)


@pytest.fixture(scope="session", autouse=True)
def isolate_test_artifacts():
    """Route all test-generated artifacts into one disposable session folder."""
    repo_root = Path(__file__).resolve().parent.parent
    artifact_root = repo_root / ".test-artifacts"

    _cleanup_repo_artifact_dirs(repo_root)
    shutil.rmtree(artifact_root, ignore_errors=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlruns_dir = artifact_root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    previous_env = {
        "TSFB_ARTIFACT_ROOT": os.environ.get("TSFB_ARTIFACT_ROOT"),
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
        "MLFLOW_DEFAULT_ARTIFACT_ROOT": os.environ.get("MLFLOW_DEFAULT_ARTIFACT_ROOT"),
    }

    os.environ["TSFB_ARTIFACT_ROOT"] = str(artifact_root)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{artifact_root / 'mlflow.db'}"
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = mlruns_dir.resolve().as_uri()

    try:
        yield artifact_root
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _cleanup_repo_artifact_dirs(repo_root)
        shutil.rmtree(artifact_root, ignore_errors=True)
