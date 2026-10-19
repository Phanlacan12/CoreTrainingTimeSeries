import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import ray
from ray import ObjectRef

from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


def is_actor() -> bool:
    """
    Determine whether the current code is running in a Ray actor context.

    Returns:
        bool: True if running in a Ray actor, False otherwise.
    """
    return ray.get_runtime_context().worker.mode == ray.WORKER_MODE


class RayActor:
    """
    Ray remote actor for executing tasks with optional environment initialization.
    """

    def __init__(self, env: Dict, initializers: Optional[List[Callable]] = None):
        """
        Initialize the RayActor.

        Args:
            env (Dict): Environment variables or configuration for the actor.
            initializers (Optional[List[Callable]]): List of initializer
                functions to run in the actor.
        """
        self._idle = True
        self._start_time: Optional[float] = None

        if initializers:
            for func in initializers:
                if func is not None and callable(func):
                    func(env)

    def run(self, fn: Callable, args: Tuple) -> Any:
        """
        Execute a function with the provided arguments inside the actor.

        Args:
            fn (Callable): The function to execute.
            args (Tuple): Arguments to pass to the function.
        Returns:
            Any: The result of the function execution.
        """
        self._start_time = time.time()
        self._idle = False
        result = fn(*args)
        self._idle = True
        return result

    def start_time(self) -> Optional[float]:
        """
        Get the start time of the last executed task if the actor is busy.

        Returns:
            Optional[float]: The start time in seconds since the epoch, or None if idle.
        """
        return None if self._idle or self._start_time is None else self._start_time


@ray.remote(max_restarts=-1)
class ObjectRefStorageActor:
    """
    Ray actor for storing and retrieving lists of Ray ObjectRefs by name.
    """

    def __init__(self):
        """
        Initialize the ObjectRefStorageActor with empty storage.
        """
        self.storage = {}

    def put(self, name: str, value: List[ObjectRef]) -> None:
        """
        Store a list of ObjectRefs under a given name.

        Args:
            name (str): The key name for the object.
            value (List[ObjectRef]): The list of Ray ObjectRefs to store.
        """
        self.storage[name] = value

    def get(self, name: str) -> Optional[List[ObjectRef]]:
        """
        Retrieve a list of ObjectRefs by name.

        Args:
            name (str): The key name for the object.
        Returns:
            Optional[List[ObjectRef]]: The stored list of ObjectRefs,
            or None if not found.
        """
        return self.storage.get(name)
