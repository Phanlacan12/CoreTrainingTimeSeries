# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import ray

from tsfb.base.schema.backend_config import RayActorPoolConfig, RayBackendConfig
from tsfb.base.utils.logging import get_logger
from tsfb.base.utils.parallel.base import BackendInterface, TaskResult
from tsfb.base.utils.parallel.ray_actor_pool import RayActorPool
from tsfb.base.utils.parallel.ray_storage import RaySharedStorage
from tsfb.base.utils.parallel.ray_utils import ObjectRefStorageActor

logger = get_logger(__name__)


class RayBackend(BackendInterface):
    """
    Backend for parallel computation using Ray.

    Manages worker pools, initializers, and CPU/GPU resource allocation.
    """

    def __init__(self, backend_config: RayBackendConfig):
        """
        Initialize the Ray backend configuration.

        Args:
            backend_config: RayBackendConfig object with initialization parameters.
        """
        self.n_cpus = (
            backend_config.n_cpus
            if backend_config.n_cpus is not None
            else os.cpu_count()
        )
        self.n_workers = (
            backend_config.n_workers
            if backend_config.n_workers is not None
            else self.n_cpus
        )
        self.gpu_devices = (
            backend_config.gpu_devices if backend_config.gpu_devices is not None else []
        )
        self.max_tasks_per_child = backend_config.max_tasks_per_child

        self.worker_initializers = (
            backend_config.worker_initializers
            if isinstance(backend_config.worker_initializers, list)
            else (
                [backend_config.worker_initializers]
                if backend_config.worker_initializers is not None
                else []
            )
        )

        self.pool: Optional[RayActorPool] = None
        self._storage: Optional[RaySharedStorage] = None
        self.initialized = False

    def init(self) -> None:  # type: ignore
        """
        Initialize the Ray backend.

        This method sets up the Ray environment,
        initializes the shared storage, and creates
        the actor pool for distributed computation. It should be
        called once before using the backend for scheduling tasks.
        """
        if self.initialized:
            return

        assert self.n_cpus is not None and self.n_workers is not None
        cpu_per_worker = self._get_cpus_per_worker(self.n_cpus, self.n_workers)
        gpu_per_worker, gpu_devices = self._get_gpus_per_worker(
            self.gpu_devices, self.n_workers
        )

        if not ray.is_initialized():
            # in the main process
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
            ray.init(num_cpus=self.n_cpus, num_gpus=len(gpu_devices))
        else:
            raise RuntimeError("init is not allowed to be called in ray actors")

        # 'put' can only be called in the main process now,
        # so the storage is safe to be accessed in multiple processes
        # pylint: disable=no-member
        remote = ObjectRefStorageActor.remote()  # type: ignore
        self._storage = RaySharedStorage(remote)

        assert self.n_workers is not None
        self.pool = RayActorPool(
            RayActorPoolConfig(
                self.n_workers,
                self.env,
                {
                    "num_cpus": cpu_per_worker,
                    "num_gpus": gpu_per_worker,
                },
                max_tasks_per_child=self.max_tasks_per_child,
                worker_initializers=self.worker_initializers,
            )
        )
        self.initialized = True

    def add_worker_initializer(self, func: Callable) -> None:
        """
        Add a function to be called when initializing each worker in the pool.

        Args:
            func (Callable): The initializer function to add.
        """
        # to_do: check if the pool is idle before updating initializer ()
        self.worker_initializers.append(func)
        assert self.pool is not None
        self.pool.worker_initializers = self.worker_initializers

    @property
    def env(self) -> Dict:
        """
        Get the environment configuration for the backend.

        Returns:
            Dict: A dictionary containing shared storage and
            other environment variables.
        """
        return {
            "storage": self.shared_storage,
        }

    def _get_cpus_per_worker(self, n_cpus: int, n_workers: int) -> float:
        """
        Calculate the number of CPUs to allocate per worker.

        Args:
            n_cpus (int): Total number of CPUs available.
            n_workers (int): Number of worker processes.
        Returns:
            float: Number of CPUs to allocate per worker.
        """
        if n_cpus > n_workers and n_cpus % n_workers != 0:
            cpus_per_worker = float(n_cpus // n_workers)
            logger.info(
                "only %d among %d cpus are used to match the number of workers",
                int(cpus_per_worker * n_workers),
                n_cpus,
            )
        else:
            cpus_per_worker = n_cpus / n_workers
        return cpus_per_worker

    def _get_gpus_per_worker(
        self, gpu_devices: List[int], n_workers: int
    ) -> Tuple[float, List[int]]:
        """
        Compute GPU devices and fraction per worker.

        Args:
            gpu_devices: List of GPU IDs available.
            n_workers: Number of workers to split GPUs across.
        Returns:
            Tuple of (GPUs per worker, list of used GPU IDs).
        """
        n_gpus = len(gpu_devices)
        if n_gpus > n_workers and n_gpus % n_workers != 0:
            gpus_per_worker = float(n_gpus // n_workers)  # ép về float luôn
            used_gpu_devices = gpu_devices[: int(gpus_per_worker * n_workers)]
            logger.info(
                "only %s gpus are used to match the number of workers", used_gpu_devices
            )
        else:
            gpus_per_worker = n_gpus / n_workers
            used_gpu_devices = gpu_devices

        return gpus_per_worker, used_gpu_devices

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> TaskResult:
        """
        Submit a function to be executed on a Ray worker.

        Args:
            fn: Function to execute remotely.
            args: Arguments to pass to the function.
            timeout: Optional timeout for execution.
        Returns:
            Result object from the scheduled execution.
        """
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        assert self.pool is not None
        return self.pool.schedule(fn, args, timeout)

    def close(self, force: bool = False) -> None:
        """
        Shut down the Ray backend and cleanup resources.

        Args:
            force: If True, do not wait for tasks to complete.
        """
        if not self.initialized or self.pool is None:
            return

        if not force:
            self.pool.wait()
        self.pool.close()
        ray.shutdown()

    @property
    def shared_storage(self) -> RaySharedStorage:
        """
        Access the shared storage used by all workers.

        Raises:
            RuntimeError: If accessed before initialization.
        """
        if self._storage is None:
            raise RuntimeError("shared_storage called before initialization")
        return self._storage

    def execute_on_workers(self, func: Callable) -> None:
        """
        Execute a function on all worker actors in the pool.

        Args:
            func (Callable): The function to execute on each worker.
            The function should accept a single argument (the environment dictionary).
        """
        if not self.pool:
            raise RuntimeError("Pool is not initialized")

        self.pool.wait()

        tasks = []
        for actor in self.pool.actors:
            tasks.append(actor.run.remote(func, (self.env,)))
        ray.wait(tasks, num_returns=len(tasks))


if __name__ == "__main__":
    logger.basicConfig(
        level=logger.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = RayBackendConfig(
        n_workers=3,
        max_tasks_per_child=1,
    )
    backend = RayBackend(config)
    backend.init()

    def sleep_func(t):
        """
        Sleep for a specified amount of time and print a message.

        Args:
            t (float): Time in seconds to sleep.

        Returns:
            float: The same time value that was passed in.
        """
        time.sleep(t)
        print(f"sleep after {t}")
        return t

    results = []
    results.append(backend.schedule(sleep_func, (10,), timeout=5))
    results.append(backend.schedule(sleep_func, (10,), timeout=20))
    results.append(backend.schedule(sleep_func, (1,), timeout=5))
    results.append(backend.schedule(sleep_func, (2,), timeout=5))
    results.append(backend.schedule(sleep_func, (3,), timeout=5))
    results.append(backend.schedule(sleep_func, (4,), timeout=5))
    results.append(backend.schedule(sleep_func, (5,), timeout=5))
    results.append(backend.schedule(sleep_func, (6,), timeout=5))

    for i, res in enumerate(results):
        try:
            print(f"{i}-th task result: {res.result()}")
        except TimeoutError:
            print(f"{i}-th task fails after timeout")

    backend.close()
    # time.sleep(100)
    # time.sleep(1)
    # pool.wait()
    # pool.close()
