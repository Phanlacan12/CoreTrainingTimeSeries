# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from tsfb.base.utils.parallel.base import BackendInterface, SharedStorage, TaskResult


class SequentialResult(TaskResult):
    """
    Concrete implementation of TaskResult for sequential (non-parallel) execution.

    Stores the result of a task and provides
        methods to set and retrieve it synchronously.
    """

    def __init__(self):
        """
        Initialize a SequentialResult with no result set.
        """
        self._result = None

    def result(self) -> Any:
        """
        Retrieve the stored result value.

        Returns:
            Any: The result value.
        """
        return self._result

    def put(self, value: Any) -> None:
        """
        Store a value as the result.

        Args:
            value (Any): The value to store as the result.
        """
        self._result = value


class SequentialSharedStorage(SharedStorage):
    """
    Concrete implementation of SharedStorage for sequential (non-parallel) execution.

    Stores variables in a local dictionary for use within a single process.
    """

    def __init__(self):
        """
        Initialize an empty storage dictionary.
        """
        self.storage = {}

    def put(self, name: str, value: Any) -> None:
        """
        Store a variable in the storage dictionary.

        Args:
            name (str): The variable name.
            value (Any): The value to store.
        """
        self.storage[name] = value

    def get(self, name: str, default_value: Any = None) -> Any:
        """
        Retrieve a variable from the storage dictionary.

        Args:
            name (str): The variable name.
            default_value (Any, optional): Value to return if the variable is not found.

        Returns:
            Any: The stored value or the default value if not found.
        """
        return self.storage.get(name, default_value)


class SequentialBackend(BackendInterface):
    """
    Backend for sequential (non-parallel) computation.

    This class provides a drop-in replacement
    for parallel backends, but executes all tasks
    sequentially in the main process. Useful for
    debugging or environments where parallelism
    is not required or not available.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs):
        """
        Initialize the SequentialBackend.

        Args:
            gpu_devices (Optional[List[int]]): List of GPU
            device IDs to use (for compatibility).
            **kwargs: Additional keyword arguments (ignored).
        """
        # pylint: disable=unused-argument
        super().__init__()
        self.gpu_devices = (
            getattr(config, "gpu_devices", []) if config is not None else []
        )
        if self.gpu_devices is None:
            self.gpu_devices = []
        self.storage: Optional[SequentialSharedStorage] = None

    def init(self) -> None:
        """
        Initialize the backend and set up shared storage.
        """
        self.storage = SequentialSharedStorage()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_devices))

    def schedule(
        self, fn: Callable, args: Tuple, timeout: float = -1
    ) -> SequentialResult:
        """
        Schedule a function for execution (runs immediately in the main process).

        Args:
            fn (Callable): The function to execute.
            args (Tuple): Arguments to pass to the function.
            timeout (float): Timeout for the task (ignored).

        Returns:
            SequentialResult: The result of the executed function.
        """
        if timeout != -1:
            warnings.warn("timeout is not supported by SequentialBackend, ignoring")
        res = SequentialResult()
        res.put(fn(*args))
        return res

    def close(self, force: bool = False) -> None:
        """
        Close the backend (no-op for sequential backend).

        Args:
            force (bool): Whether to forcefully close (ignored).
        """
        # pylint: disable=unused-argument
        return None

    @property
    def shared_storage(self) -> SharedStorage:
        """
        Get the shared storage instance.

        Returns:
            SharedStorage: The shared storage instance.
        """
        if self.storage is None:
            raise RuntimeError("Backend not initialized")
        return self.storage

    @property
    def env(self) -> Dict:
        """
        Get the environment dictionary for the backend.

        Returns:
            Dict: The environment dictionary containing shared storage.
        """
        return {
            "storage": self.shared_storage,
        }

    def execute_on_workers(self, func: Callable) -> None:
        """
        Execute a function in the main process (
        for compatibility with parallel backends).

        Args:
            func (Callable): The function to execute.
        """
        func(self.env)

    def add_worker_initializer(self, func: Callable) -> None:
        """
        Add a worker initializer function (no-op for sequential backend).

        Args:
            func (Callable): The initializer function to add.
        """
        # pylint: disable=unused-argument
        return None
