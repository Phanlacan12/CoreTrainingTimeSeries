import itertools
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError

from tsfb.base.schema.backend_config import RayActorPoolConfig
from tsfb.base.utils.logging import get_logger
from tsfb.base.utils.parallel.ray_task import RayResult, RayTask
from tsfb.base.utils.parallel.ray_utils import RayActor

logger = get_logger(__name__)


class RayActorPool:
    """
    Ray Actor Resource Pool with task timeout support.
    """

    def __init__(self, pool_config: RayActorPoolConfig):
        """
        Initialize the RayActorPool.

        Args:
            config: Configuration object for RayActorPool.
        """
        self.env = pool_config.env
        self.per_worker_resources = pool_config.per_worker_resources or {}
        self.max_tasks_per_child = pool_config.max_tasks_per_child
        self.worker_initializers = pool_config.worker_initializers
        n_workers = pool_config.n_workers

        self.actor_class = ray.remote(
            max_restarts=0,
            num_cpus=self.per_worker_resources.get("num_cpus", 1),
            num_gpus=self.per_worker_resources.get("num_gpus", 0),
        )(RayActor)
        self.actors = [self._new_actor() for _ in range(n_workers)]

        # Main thread state
        self._task_counter = itertools.count()

        # Loop thread state
        self._task_info: Dict[Any, Any] = {}
        self._ray_task_to_id: Dict[Any, Any] = {}
        self._active_tasks: List[Any] = []
        self._idle_actors = list(range(len(self.actors)))
        self._restarting_actor_pool: Dict[Any, Any] = {}
        self._actor_tasks = [0] * len(self.actors)

        # Communication
        self._is_closed = False
        self._idle_event = threading.Event()
        self._pending_queue: queue.Queue = queue.Queue(maxsize=1000000)

        self._main_loop_thread = threading.Thread(target=self._main_loop)
        self._main_loop_thread.start()

    def _new_actor(self) -> ActorHandle:
        """
        Create a new Ray actor with the given environment and initializer.

        Returns:
            A handle to the newly created Ray actor.
        """
        max_concurrency = (
            100
            if sys.platform == "win32"
            and self.per_worker_resources.get("num_gpus", 0) > 0
            else 2
        )
        handle = self.actor_class.options(max_concurrency=max_concurrency).remote(
            self.env,
            self.worker_initializers,
        )  # type: ignore[arg-type,call-arg]

        return cast(ActorHandle, handle)

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        """
        Schedule a task to be executed by an available actor.

        Args:
            fn: The function to execute remotely.
            args: Arguments to pass to the function.
            timeout: Optional timeout in seconds. Default is -1 (no timeout).

        Returns:
            RayResult: An object for collecting the result or exception.
        """
        self._idle_event.clear()
        task_id = next(self._task_counter)
        result = RayResult(threading.Event())
        self._pending_queue.put((fn, args, timeout, task_id, result), block=True)
        return result

    def _handle_ready_tasks(self, tasks: List) -> None:
        """
        Handle tasks that have completed execution.

        Args:
            tasks (List): List of Ray task objects that are ready.
        """
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            try:
                task_info.result.put(ray.get(task_obj))
            except RayActorError as e:
                logger.info(
                    "task %d died unexpectedly on actor %d: %s",
                    task_id,
                    task_info.actor_id,
                    e,
                )
                task_info.result.put(RuntimeError(f"task died unexpectedly: {e}"))
                self._restart_actor(task_info.actor_id)
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
                continue

            self._actor_tasks[task_info.actor_id] += 1
            if (
                self.max_tasks_per_child is not None
                and self._actor_tasks[task_info.actor_id] >= self.max_tasks_per_child
            ):
                logger.info(
                    "max_tasks_per_child reached in actor %s, restarting",
                    task_info.actor_id,
                )
                self._restart_actor(task_info.actor_id)
            else:
                self._idle_actors.append(task_info.actor_id)
            del self._task_info[task_id]
            del self._ray_task_to_id[task_obj]

    def _get_duration(self, task_info: RayTask) -> Optional[float]:
        """
        Get the duration (in seconds) that a task has been running.

        Args:
            task_info (RayTask): The task information object.

        Returns:
            Optional[float]: Duration in seconds, or None if not available.
        """
        if task_info.actor_id is None:
            logger.warning("actor_id is None, cannot get duration")
            return None

        # If we haven't recorded the start_time, try to fetch from actor
        if task_info.start_time is None:
            try:
                result = ray.get(self.actors[task_info.actor_id].start_time.remote())
                if result is None:
                    logger.debug("start_time is None (actor probably idle)")
                    task_info.start_time = None
                elif isinstance(result, (int, float)):
                    task_info.start_time = float(result)
                else:
                    logger.warning(
                        "Unexpected type for start_time from actor %d: %s",
                        task_info.actor_id,
                        type(result),
                    )
                    task_info.start_time = None
            except RayActorError as e:
                logger.info(
                    "actor %d died unexpectedly while getting start_time: %s",
                    task_info.actor_id,
                    e,
                )
                return None

        if task_info.start_time is None:
            # Still None after fetching: no duration info
            return -1.0

        assert isinstance(task_info.start_time, float)
        return float(time.time() - task_info.start_time)

    def _handle_unfinished_tasks(self, tasks: List) -> None:
        new_active_tasks = []
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            duration = self._get_duration(task_info)
            if duration is None or (
                task_info.timeout is not None and 0 < task_info.timeout < duration
            ):
                if duration is not None:
                    logger.info(
                        "actor %d killed after timeout %f",
                        task_info.actor_id,
                        task_info.timeout,
                    )
                self._restart_actor(task_info.actor_id)
                task_info.result.put(
                    TimeoutError(f"time limit exceeded: {task_info.timeout}")
                )
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
            else:
                new_active_tasks.append(task_obj)
        self._active_tasks = new_active_tasks

    def _restart_actor(self, actor_id: int) -> None:
        """
        Restart a Ray actor by killing the current one and creating a new one.

        Args:
            actor_id (int): The ID of the actor to restart.
        """
        cur_actor = self.actors[actor_id]
        ray.kill(cur_actor, no_restart=True)
        del cur_actor
        self.actors[actor_id] = self._new_actor()
        self._actor_tasks[actor_id] = 0
        self._restarting_actor_pool[actor_id] = time.time()

    def _check_restarting_actors(self):
        """
        Check and update the status of actors that are being restarted.
        """
        new_restarting_pool = {}
        for actor_id, restart_time in self._restarting_actor_pool.items():
            if time.time() - restart_time > 5:
                ready_tasks = ray.wait(
                    [self.actors[actor_id].start_time.remote()], timeout=0.5
                )[0]
                if ready_tasks:
                    logger.debug("restarted actor %d is now ready", actor_id)
                    self._idle_actors.append(actor_id)
                    continue

                logger.debug(
                    "restarted actor %d is not ready, resetting timer", actor_id
                )
                self._restarting_actor_pool[actor_id] = time.time()

            new_restarting_pool[actor_id] = restart_time

        self._restarting_actor_pool = new_restarting_pool

    def _main_loop(self) -> None:
        """
        Main loop for managing task scheduling, actor assignment, and task completion.

        This loop continuously checks for ready and unfinished
        tasks, assigns new tasks to idle actors,
        and handles actor restarts and timeouts.
        It runs in a dedicated thread and is responsible for
        orchestrating the Ray actor pool's task execution lifecycle.
        """
        while not self._is_closed:
            self._check_restarting_actors()

            logger.debug(
                "%d active tasks, %d idle actors, %d restarting actors",
                len(self._active_tasks),
                len(self._idle_actors),
                len(self._restarting_actor_pool),
            )

            if not self._active_tasks and self._pending_queue.empty():
                self._idle_event.set()
                time.sleep(1)
                continue

            if self._active_tasks:
                ready_tasks, unfinished_tasks = ray.wait(self._active_tasks, timeout=1)
                self._handle_ready_tasks(ready_tasks)
                self._handle_unfinished_tasks(unfinished_tasks)
            else:
                time.sleep(1)

            while self._idle_actors and not self._pending_queue.empty():
                fn, args, timeout, task_id, result = self._pending_queue.get_nowait()
                cur_actor = self._idle_actors.pop()
                task_obj = self.actors[cur_actor].run.remote(fn, args)
                self._task_info[task_id] = RayTask(
                    result=result, actor_id=cur_actor, timeout=timeout
                )
                self._ray_task_to_id[task_obj] = task_id
                self._active_tasks.append(task_obj)
                logger.debug("task %d assigned to actor %d", task_id, cur_actor)

    def wait(self) -> None:
        """
        Block until all pending and active tasks are completed and all
        restarting actors are ready.
        """
        if self._is_closed:
            return
        if self._pending_queue.empty() and not self._active_tasks:
            return
        self._idle_event.clear()
        self._idle_event.wait()
        while self._restarting_actor_pool:
            time.sleep(1)

    def close(self) -> None:
        """
        Close the actor pool, terminate all actors, and join the main loop thread.
        """
        self._is_closed = True
        for actor in self.actors:
            ray.kill(actor)
        self._main_loop_thread.join()
