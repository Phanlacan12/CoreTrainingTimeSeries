# -*- coding: utf-8 -*-
import importlib
from typing import Any, Callable, Dict, List, Optional, Union

from tsfb.base.models.darts.darts_factory import get_adapter_mapping
from tsfb.base.models.foundation.registry import get_foundation_spec
from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)

_FOUNDATION_PREFIX = "foundation."


def _load_foundation_model(model_config: Dict) -> Callable:
    """
    Instantiate a foundation model adapter from the registry.

    model_name must be of the form "foundation.<key>" where <key> is a
    registered entry in FOUNDATION_MODEL_REGISTRY (e.g. "foundation.chronos").
    model_args is passed as keyword arguments to the adapter constructor.
    """

    key = model_config["model_name"][len(_FOUNDATION_PREFIX) :]
    spec = get_foundation_spec(key)

    package_name, attr_name = spec.adapter_class.rsplit(".", 1)
    module = importlib.import_module(package_name)
    adapter_cls = getattr(module, attr_name)

    model_args = model_config.get("model_args", {})
    if not model_args.get("pretrained_model_id"):
        model_args = {"pretrained_model_id": spec.default_model_id, **model_args}

    def factory(**kwargs):
        merged = {**model_args, **kwargs}
        return adapter_cls(**merged)

    factory_any: Any = factory
    factory_any.model_name = key
    return factory


def _import_attribute(attr_path: str) -> Any:
    """
    import attribute according to a fully qualified path.

    :param attr_path: A dot-separated path.
    :return: If the target attribute exists,
    the attribute is returned, otherwise return None.
    """
    package_name, name = attr_path.rsplit(".", 1)
    package = importlib.import_module(package_name)
    return getattr(package, name, None)


def import_model_info(model_path: str) -> Union[Dict, Callable]:
    """
    Import model information.

    We first clarify some concepts before defining model information:

    - required hyperparameters: This is a specially
    designed mechanism to enable models to relinquish the settings
      of some hyperparameters to the benchmark.
      For example, if a model cannot automatically decide the best input window size
      (corresponding hyperparameter `input_window_size`),
      it can leave the decision to the benchmark, so that
      the benchmark is able to use a globally recommended
      setting (corresponding hyperparameter `input_chunk_length`)
      to produce a fair comparison between different models;.
      In this example, to enable this mechanism properly,
      the model is required to provide a
      `required_hyper_params` field in dictionary
      `{"input_window_size": "input_chunk_length"}`.

    Model information should be either:

    - A dictionary containing these fields:

        - model_factory: Callable. A callable that accepts hyperparameters as kwargs;
        - model_hyper_params: Dictionary,
            optional; A dictionary containing hyperparameters for the model.
            These hyperparameters overwrite the ones
            specified by recommended hyperparameters;
        - required_hyper_params: Dictionary, optional;
            A dictionary of hyperparameters to be filled
            by the benchmark, in format `{model_param_name: std_param_name}`.
        - model_name: str, optional; The name of
            the model that is recorded in the output logs.

    - A callable that returns an instance compatible
        with :class:`ModelBase` interface when called with
        hyperparameters as keyword arguments.
        This callable may optionally support the following features:

        - attribute required_hyper_params: Dictionary,
            optional; A dictionary of hyperparameters to be
            filled by the benchmark, in format `{model_param_name: std_param_name}`.

    :param model_path: The fully qualified path to the model information.
    :return: The imported model information.
    """
    model_info = _import_attribute(model_path)

    if not isinstance(model_info, dict) and not callable(model_info):
        raise ValueError(
            f"Unsupported model info with type {type(model_info).__name__}"
        )

    return model_info


def get_model_info(model_config: Dict) -> Optional[Union[dict, Callable]]:
    """
    Obtain model information based on model configuration.

    :param model_config: A dictionary that contains
        model configuration information. The supported fields are:

        - model_name: str. The path to the model information,
            the following paths are searched in order to
          find the model information:

            - `{model_name[7:]}` if model_name.startswith("global.")
            - `ts_benchmark.baselines.{model_name}`
            - `{model_name}`

        - adapter: str, optional. The adapter name to wrap the found model information.
          Must be one of the adapters defined in :mod:`ts_benchmark.baselines.__init__`;

    :return: The model information corresponding to the config.
    :raises ImportError: If the specified model package cannot be imported.
    :raises AttributeError: If the specified `model_name`
        cannot be found in the imported module.
    """
    if model_config["model_name"].startswith(_FOUNDATION_PREFIX):
        logger.info("Loading foundation model: %s", model_config["model_name"])
        return _load_foundation_model(model_config)

    model_name_candidates = [
        (
            model_config["model_name"][7:]
            if model_config["model_name"].startswith("global.")
            else None
        ),
        "tsfb.base.models." + model_config["model_name"],
        model_config["model_name"],
    ]
    model_name_candidates = list(filter(None, model_name_candidates))

    model_info = None
    for i, m in enumerate(model_name_candidates):
        try:
            logger.info("Trying %s", m)
            model_info = import_model_info(m)
            break
        except (ImportError, AttributeError) as e:
            nxt = (
                f" -> next: {model_name_candidates[i + 1]}"
                if i + 1 < len(model_name_candidates)
                else " (no more)"
            )
            logger.warning("Load %s failed: %s%s", m, e, nxt)

    if isinstance(model_info, dict) or callable(model_info):
        model_name_key = model_config["model_name"].split(".")[-1]
        adapter_mapping = get_adapter_mapping()
        if model_name_key in adapter_mapping:
            logger.info("Using auto adapter for model: %s", model_name_key)
            model_info = adapter_mapping[model_name_key](model_info)

    return model_info


def get_model_hyper_params(
    recommend_model_hyper_params: Dict, required_hyper_params: Dict, model_config: Dict
) -> Dict:
    """
    Obtain the hyperparameters of the model.

    The hyperparameter dictionary is constructed following these steps:

    - Fill in the recommended hyperparameters;
    - Update the hyperparameters with those specified in the model_config;

    :param recommend_model_hyper_params: A dictionary of
        hyperparameters recommended by the benchmark.
    :param required_hyper_params: A dictionary of
        hyperparameters to be filled by the benchmark,
        in format `{model_param_name: std_param_name}`.
        Please refer to :func:`import_model_info` for
        details about this argument.
    :param model_config: Model configuration, the supported fields are:

        - model_hyper_params: dictionary, optional;
            This dictionary specifies the hyperparameters used
            in the corresponding model;

    :return: The constructed model hyperparameter dictionary.
    :raises ValueError: If there are unfilled hyperparameters.
    """
    model_hyper_params = {
        arg_name: recommend_model_hyper_params[arg_std_name]
        for arg_name, arg_std_name in required_hyper_params.items()
        if arg_std_name in recommend_model_hyper_params
    }
    model_hyper_params.update(model_config.get("model_hyper_params", {}))

    lr = model_hyper_params.pop("learning_rate", None)
    if lr is not None:
        if "optimizer_kwargs" not in model_hyper_params:
            model_hyper_params["optimizer_kwargs"] = {}
        model_hyper_params["optimizer_kwargs"]["lr"] = lr

    missing_hp = set(required_hyper_params) - set(model_hyper_params)
    if missing_hp:
        raise ValueError(f"These hyper parameters are missing : {missing_hp}")
    return model_hyper_params


class ModelFactory:
    """
    Model factory, the standard type to instantiate models in the pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Callable,
        model_hyper_params: dict,
    ):
        """
        Initialize the ModelFactory object.

        :param model_name: Model name.
        :param model_factory: A model factory (classes or
            factory functions) used to create model instances.
        :param model_hyper_params: The hyperparameter dictionary
            used to instantiate the model instance.
        """
        self.model_name = model_name
        self.model_factory = model_factory
        self.model_hyper_params = model_hyper_params

    def __call__(self) -> Any:
        """
        Instantiate the model.

        :return: A model instance that is compatible
            with the :class:`ModelBase` interface.
        """
        return self.model_factory(**self.model_hyper_params)


def get_models(all_model_config: Dict) -> List[ModelFactory]:
    """
    Obtain a list of ModelFactory objects based on model configuration.

    :param all_model_config: A dictionary that contains all
        model configuration information, supported fields are:

        - models: list. A list of model information, where each item is a dictionary.
          The supported fields in each dictionary are:

            - model_name: str. The path to the model information.
                Please refer to :func:`get_model_info` for
                the details about the model searching strategy;
            - adapter: str, optional. The adapter name to
                wrap the found model information.
                Must be one of the adapters defined
                in :mod:`ts_benchmark.baselines.__init__`;

        - recommend_model_hyper_params: dictionary, optional;
            A dictionary of globally recommended hyperparameters
            that the benchmark supplies to all models;

    :return: List of model factories used to instantiate different models.
    """
    model_factory_list = []  # Store a list of model factories
    # Traverse each model configuration
    for model_config in all_model_config["models"]:
        model_info = get_model_info(model_config)  # Obtain model information
        fallback_model_name = model_config["model_name"].split(".")[-1]

        # Analyze model information
        if isinstance(model_info, dict):
            model_factory = model_info.get("model_factory")
            if model_factory is None:
                raise ValueError("model_factory is none")
            required_hyper_params = model_info.get("required_hyper_params", {})
            model_name = model_info.get("model_name", fallback_model_name)
        elif callable(model_info):
            model_factory = model_info
            required_hyper_params = {}
            if hasattr(model_factory, "required_hyper_params"):
                required_hyper_params = model_factory.required_hyper_params()
            model_name = fallback_model_name
        else:
            raise ValueError(f"Unexpected model info type {type(model_info).__name__}")

        model_hyper_params = get_model_hyper_params(
            all_model_config.get("recommend_model_hyper_params", {}),
            required_hyper_params,
            model_config,
        )
        # Add Model Factory to List
        model_factory_list.append(
            ModelFactory(model_name, model_factory, model_hyper_params)
        )
    return model_factory_list


def get_single_model(model_name: str, model_hyper_params: Dict) -> Any:
    """
    Instantiate a single model from its name and hyperparameter configuration.

    This function wraps the model instantiation process using the
    `get_models()` utility. It builds a temporary model configuration
    structure with a single model, then returns the instantiated model.

    Parameters
    ----------
    model_name : str
        The fully qualified name or module path of the model to import and instantiate.
    model_hyper_params : Dict
        A dictionary of hyperparameters to initialize the model with.

    Returns
    -------
    Any
        A model instance compatible with the `ModelBase` interface.

    Raises
    ------
    ValueError
        If the model cannot be instantiated properly.
    """
    model_config = {
        "models": [
            {
                "model_name": model_name,
                "model_hyper_params": model_hyper_params,
            }
        ]
    }
    model_factories = get_models(model_config)
    return model_factories[0]()
