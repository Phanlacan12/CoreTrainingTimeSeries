from typing import Any

import ray

from tsfb.base.utils.logging import get_logger
from tsfb.base.utils.parallel.base import SharedStorage
from tsfb.base.utils.parallel.ray_utils import is_actor

logger = get_logger(__name__)


class RaySharedStorage(SharedStorage):
    """
    Shared storage implementation using Ray actors
    for distributed object reference management.
    """

    def __init__(self, object_ref_actor):
        """
        Initialize RaySharedStorage with a reference to an ObjectRefStorageActor.

        Args:
            object_ref_actor: The Ray actor for object reference storage.
        """
        self.object_ref_actor = object_ref_actor

    def put(self, name: str, value: Any) -> None:
        """
        Store a value in shared storage by serializing it as a Ray ObjectRef.

        Args:
            name (str): The key name for the object.
            value (Any): The value to store.
        """
        if is_actor():
            raise RuntimeError("put is not supported to be called by actors")
        obj_ref = ray.put(value)
        ray.get(self.object_ref_actor.put.remote(name, [obj_ref]))

    def get(self, name: str, default_value: Any = None) -> Any:
        """
        Retrieve a value from shared storage by name.

        Args:
            name (str): The key name for the object.
            default_value (Any, optional): Value to return if the key is not found.
        Returns:
            Any: The retrieved value, or default_value if not found.
        """
        obj_ref = ray.get(self.object_ref_actor.get.remote(name))
        if obj_ref is None:
            logger.info("data '%s' does not exist in shared storage", name)
            return default_value
        return ray.get(obj_ref[0])
