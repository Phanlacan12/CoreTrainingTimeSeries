from typing import Callable, Optional, Tuple

from tsfb.base.schema.backend_config import ParallelBackendConfig
from tsfb.base.utils.design_pattern import SingletonMeta
from tsfb.base.utils.parallel.base import BackendInterface, SharedStorage, TaskResult
from tsfb.base.utils.parallel.ray_backend import RayBackend
from tsfb.base.utils.parallel.sequential_backend import SequentialBackend

__all__ = ["ParallelBackend", "SharedStorage"]


class ParallelBackend(metaclass=SingletonMeta):
    """
    Singleton manager for parallel computation backends.

    This class provides a unified interface to initialize and manage different parallel
    backends (e.g., Ray, sequential). It supports backend selection,
    resource configuration,
    and task scheduling for distributed computation.
    """

    BACKEND_DICT = {
        "ray": RayBackend,
        "sequential": SequentialBackend,
    }

    def __init__(self) -> None:
        """
        Initialize the ParallelBackend manager with default (uninitialized) backend.
        """
        self.backend: Optional[BackendInterface] = None
        self.default_timeout: Optional[float] = None

    def init(self, config: ParallelBackendConfig) -> None:
        """
        Initialize the selected parallel backend using the provided configuration.

        Args:
            config (ParallelBackendConfig): Configuration for the backend to initialize.

        Raises:
            ValueError: If the backend name is unknown.
            RuntimeError: If the backend is already initialized.
        """
        if config.backend not in self.BACKEND_DICT:
            raise ValueError(f"Unknown backend name {config.backend}")
        if self.backend is not None:
            raise RuntimeError("Please close the backend before re-initializing")

        self.backend = self.BACKEND_DICT[config.backend](config)
        self.backend.init()
        self.default_timeout = config.default_timeout

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> TaskResult:
        """
        Schedule a task to run in the backend.

        Args:
            fn (Callable): The function to be executed.
            args (Tuple): Arguments to pass to the function.
            timeout (Optional[float]): Timeout in seconds for task execution.

        Returns:
            TaskResult: The result wrapper of the scheduled task.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        if self.backend is None:
            raise RuntimeError(
                "Please initialize parallel backend before calling schedule"
            )
        if timeout is None:
            timeout = self.default_timeout
        return self.backend.schedule(fn, args, timeout)

    def close(self, force: bool = False) -> None:
        """
        Close and clean up the backend, terminating all workers.

        Args:
            force (bool): Whether to force termination immediately.
        """
        if self.backend is not None:
            self.backend.close(force)
            self.backend = None

    @property
    def shared_storage(self) -> SharedStorage:
        """
        Access shared storage used across workers in the backend.

        Returns:
            SharedStorage: Shared memory or storage used by backend.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")
        return self.backend.shared_storage

    def add_worker_initializer(self, func: Callable) -> None:
        """
        Add a worker initializer to be called on each worker process.

        Args:
            func (Callable): The function to run on worker startup.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")
        self.backend.add_worker_initializer(func)

    def execute_on_workers(self, func: Callable) -> None:
        """
        Execute a function immediately on all worker processes.

        Args:
            func (Callable): The function to run on all workers.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")
        self.backend.execute_on_workers(func)
