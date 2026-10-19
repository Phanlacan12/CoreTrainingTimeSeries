import threading
from typing import Any, Optional

from tsfb.base.utils.parallel.base import TaskResult


class RayResult(TaskResult):
    """
    Result wrapper for Ray tasks, supporting asynchronous
    result retrieval and exception handling.
    """

    __slots__ = ["_event", "_result"]

    def __init__(self, event: threading.Event):
        """
        Initialize a RayResult with a threading event.

        Args:
            event (threading.Event): Event to signal result availability.
        """
        self._event = event
        self._result = None

    def put(self, value: Any) -> None:
        """
        Store the result value and signal completion.

        Args:
            value (Any): The result value or exception to store.
        """
        self._result = value
        self._event.set()

    def result(self) -> Any:
        """
        Wait for and return the result value, raising if it is an exception.

        Returns:
            Any: The result value.
        Raises:
            Exception: If the result is an exception, it is raised.
        """
        self._event.wait()
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class RayTask:
    """
    Data structure for tracking Ray task state and metadata.
    """

    __slots__ = ["result", "actor_id", "timeout", "start_time"]

    def __init__(
        self, result: Any = None, actor_id: Optional[int] = None, timeout: float = -1
    ):
        """
        Initialize a RayTask.

        Args:
            result (Any, optional): The result object for the task.
            actor_id (Optional[int], optional): The actor ID assigned to the task.
            timeout (float, optional): Timeout for the task in seconds.
        """
        self.result = result
        self.actor_id = actor_id
        self.timeout = timeout
        self.start_time: Optional[float] = None
