import contextlib
import io
import sys
from unittest.mock import MagicMock, patch

import pytest

from tsfb.base.utils.parallel.sequential_backend import SequentialBackend


# ==========================================
# Suppress LightGBM warning during tests
# ==========================================
@contextlib.contextmanager
def suppress_lightgbm_warning():
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr


@pytest.fixture(autouse=True)
def suppress_lgbm_warning_fixture():
    with suppress_lightgbm_warning():
        yield


# =========================
# Tests for SequentialBackend
# =========================


def test_sequential_backend_schedule():
    backend = SequentialBackend()
    backend.init()

    def add(x, y):
        return x + y

    result = backend.schedule(add, (2, 3))
    assert result.result() == 5


def test_sequential_backend_shared_storage():
    backend = SequentialBackend()
    backend.init()
    storage = backend.shared_storage
    storage.put("foo", 123)
    assert storage.get("foo") == 123
    assert storage.get("bar", "default") == "default"


def test_sequential_backend_execute_on_workers():
    backend = SequentialBackend()
    backend.init()
    called = {}

    def func(env):
        called["ok"] = env["storage"] is backend.shared_storage

    backend.execute_on_workers(func)
    assert called["ok"]


def test_sequential_backend_add_worker_initializer():
    backend = SequentialBackend()
    backend.init()
    backend.add_worker_initializer(lambda env: None)  # Should not raise


# =========================
# Tests for RayBackend (mocked)
# =========================


def test_ray_backend_schedule_and_storage():
    with patch("tsfb.base.utils.parallel.ray_backend.ray") as ray_mock:
        # Setup ray mock
        ray_mock.is_initialized.return_value = False
        ray_mock.init.return_value = None
        ray_mock.put.side_effect = lambda x: x
        ray_mock.get.side_effect = lambda x: x
        ray_mock.remote.side_effect = lambda *a, **k: MagicMock()
        ray_mock.WORKER_MODE = 1
        ray_mock.get_runtime_context.return_value = MagicMock(worker=MagicMock(mode=1))

        # Import RayBackend after mocking
        from tsfb.base.schema.backend_config import RayBackendConfig
        from tsfb.base.utils.parallel.ray_backend import RayBackend

        backend = RayBackend(RayBackendConfig(n_workers=1))
        backend.init()

        # Mock pool.schedule
        backend.pool.schedule = MagicMock()
        backend.pool.schedule.return_value.result = lambda: 42

        result = backend.schedule(lambda x: x, (1,))
        assert result.result() == 42

        # Mock shared_storage
        backend._storage.put = MagicMock()
        backend._storage.get = MagicMock(return_value=99)

        backend.shared_storage.put("foo", 99)
        assert backend.shared_storage.get("foo") == 99

        backend.close(force=True)
