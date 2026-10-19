from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union


@dataclass
class RayBackendConfig:
    """
    Configuration class for the Ray parallel backend.

    Attributes:
        n_workers (Optional[int]): Number of Ray actors (workers) to launch.
        n_cpus (Optional[int]): Total number of CPUs to allocate for the backend.
        gpu_devices (Optional[List[int]]): List of GPU
            device indices available for assignment.
        max_tasks_per_child (Optional[int]): Maximum number
            of tasks a worker should execute before being restarted.
        worker_initializers (Optional[Union[List[Callable], Callable]]):
            One or more callables to initialize each worker.
    """

    n_workers: Optional[int] = None
    n_cpus: Optional[int] = None
    gpu_devices: Optional[List[int]] = None
    max_tasks_per_child: Optional[int] = None
    worker_initializers: Optional[Union[List[Callable], Callable]] = None


@dataclass
class RayActorPoolConfig:
    """
    Configuration class for initializing the Ray actor pool.

    Attributes:
        n_workers (int): Number of actor workers to be created.
        env (Dict): Environment variables or context to be shared with each actor.
        per_worker_resources (Optional[Dict]): Resource allocation
            per worker, e.g., {'num_cpus': 1, 'num_gpus': 0}.
        max_tasks_per_child (Optional[int]): Maximum number
            of tasks a worker should execute before restart.
        worker_initializers (Optional[List[Callable]]): List of functions
            to be called during actor initialization.
    """

    n_workers: int
    env: Dict
    per_worker_resources: Optional[Dict] = None
    max_tasks_per_child: Optional[int] = None
    worker_initializers: Optional[List[Callable]] = None


@dataclass
class ParallelBackendConfig:
    """
    Configuration class for selecting and initializing a parallel backend.

    Attributes:
        backend (str): Name of the backend to use ("ray" or "sequential").
        n_workers (Optional[int]): Number of parallel workers to use.
        n_cpus (Optional[int]): Number of CPUs to
            allocate.
        gpu_devices (Optional[List[int]]): List of GPU
            indices available for computation.
        default_timeout (float): Default timeout in
            seconds for tasks (-1 means no timeout).
        max_tasks_per_child (Optional[int]): Number of tasks
            per worker before restart (for Ray).
        worker_initializers (Optional[Union[List[Callable], Callable]]):
            One or more functions to initialize each worker.
    """

    backend: str = "ray"
    n_workers: Optional[int] = None
    n_cpus: Optional[int] = None
    gpu_devices: Optional[List[int]] = None
    default_timeout: float = -1.0
    max_tasks_per_child: Optional[int] = None
    worker_initializers: Optional[Union[List[Callable], Callable]] = None
