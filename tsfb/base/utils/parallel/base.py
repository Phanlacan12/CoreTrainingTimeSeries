"""
Abstract interfaces for asynchronous task execution and shared storage backends.

This module defines the core interfaces for building backends that execute tasks
(either sequentially or in parallel/distributed environments) and share data
between workers. These interfaces are intended to be subclassed and implemented
by concrete backends such as multiprocessing, Ray, or sequential execution engines.
"""

from __future__ import absolute_import

import abc
from typing import Any, Callable, Tuple


class TaskResult(metaclass=abc.ABCMeta):
    """
    Abstract base class representing the result of an asynchronous task.

    This interface provides two core methods:
    - `put`: to set the result (typically called by the backend).
    - `result`: to retrieve the result (possibly blocking until available).

    Intended to be implemented by task/result wrappers in specific backends
    (e.g., Future-like objects, queues, etc.).
    """

    @abc.abstractmethod
    def result(self) -> Any:
        """
        Block until the task result is available, then return it.

        Returns:
            Any: The final result of the task, set via `put()`.
        """

    @abc.abstractmethod
    def put(self, value: Any) -> None:
        """
        Store the result of the task (usually set internally by the backend).

        Args:
            value (Any): The result value to store.
        """


class SharedStorage(metaclass=abc.ABCMeta):
    """
    Abstract base class for shared storage between workers.

    Shared storage enables communication and
    variable exchange between processes or nodes
    during parallel or distributed execution. Typical
    implementations may use dictionaries,
    Redis, or shared memory.

    The storage is key-value-based and must support default fallback.
    """

    @abc.abstractmethod
    def put(self, name: str, value: Any) -> None:
        """
        Store a variable in the shared storage.

        Args:
            name (str): The key/name of the variable.
            value (Any): The value to store.
        """

    @abc.abstractmethod
    def get(self, name: str, default_value: Any = None) -> Any:
        """
        Retrieve a variable from shared storage by name.

        Args:
            name (str): The key/name of the variable.
            default_value (Any, optional): The fallback value if key is not found.

        Returns:
            Any: The value associated with the key, or default if not found.
        """


class BackendInterface(metaclass=abc.ABCMeta):
    """
    Abstract interface for backend execution engines.

    A backend is responsible for:
    - initializing resources (e.g., workers, memory),
    - scheduling functions for execution (possibly in parallel),
    - providing a shared storage for communication between workers,
    - optionally executing initialization logic across workers.

    Typical implementations include:
    - SequentialBackend (for single-process debug/test)
    - RayBackend (for distributed async execution)
    """

    @abc.abstractmethod
    def init(self) -> None:
        """
        Initialize backend resources and setup shared storage or worker pool.
        """

    @abc.abstractmethod
    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> TaskResult:
        """
        Schedule a task for execution.

        Args:
            fn (Callable): The function to execute.
            args (Tuple): Arguments to pass to the function.
            timeout (float, optional): Max time (in seconds) to wait for completion.

        Returns:
            TaskResult: An object representing the future result of the task.
        """

    @abc.abstractmethod
    def close(self, force: bool = False) -> None:
        """
        Shutdown the backend and clean up resources.

        Args:
            force (bool): Whether to forcibly stop all tasks/workers.
        """

    @property
    @abc.abstractmethod
    def shared_storage(self) -> SharedStorage:
        """
        Access the backend's shared storage interface.

        Returns:
            SharedStorage: An object for storing/retrieving global variables.
        """

    @abc.abstractmethod
    def add_worker_initializer(self, func: Callable) -> None:
        """
        Register an initializer function to be run on all workers.

        Args:
            func (Callable): A function that prepares environment on each worker.
        """

    @abc.abstractmethod
    def execute_on_workers(self, func: Callable) -> None:
        """
        Execute a given function on all workers (usually used for broadcast or setup).

        Args:
            func (Callable): The function to be executed on each worker.
        """
