import logging
from typing import Any, Callable, Dict, Optional, Tuple

from .base.approaches.base import ForecastingApproach
from .base.approaches.default import DefaultApproach
from .base.approaches.hybrid import HybridApproach
from .base.approaches.univariate import UnivariateToMultivariate
from .base.models.model_loader import get_models, get_single_model


def build_approaches(
    cfg: dict, strategy_args: dict, logger: logging.Logger
) -> Dict[str, ForecastingApproach]:
    """Build forecasting approaches from model configuration."""
    logger.info("Building approaches…")
    model_cfg: Dict[str, Dict[str, Any]] = cfg.get("model", {}) or {}
    approaches: Dict[str, ForecastingApproach] = {}

    for name, conf in model_cfg.items():
        atype = conf.get("approach") or ("hybrid" if "group" in conf else "default")
        conf_copy = conf.copy()  # Avoid cell-var-from-loop warning

        def _make_factory(conf_=conf_copy) -> Optional[Callable[[], Any]]:
            f = get_models({"models": [conf_]})
            return f[0] if f else None

        app: Optional[ForecastingApproach] = None
        if atype == "default":
            factory = _make_factory() or (_ for _ in ()).throw(
                ValueError(f"Missing model factory for {name}")
            )
            app = DefaultApproach(model_factory=factory)

        elif atype == "univariate_per_series":
            factory = _make_factory() or (_ for _ in ()).throw(
                ValueError(f"Missing model factory for {name}")
            )
            app = UnivariateToMultivariate(model_factory=factory)

        elif atype == "hybrid":
            conf = {
                **conf,
                "data_module_config": strategy_args.get("data_module_config", {}),
            }
            app = HybridApproach(config=conf)

        else:
            raise ValueError(f"Unknown approach type '{atype}' for '{name}'")

        app.config, app.name = conf, name
        approaches[name] = app
        logger.info("  • Built %s (%s)", app.__class__.__name__, name)

    return approaches


def _conf_from_best(best: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model configuration from best['model_params']."""
    if not best:
        raise ValueError("best is None or empty")
    conf = dict(best.get("model_params", {}))
    conf.setdefault("approach", "default")
    conf["name"] = best.get("name", "best_model")
    return conf


def build_model(best: Dict[str, Any]):
    """Initialize a single model instance (ModelBase) from 'best'."""
    return get_models({"models": [_conf_from_best(best)]})[0]()


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger:
        return logger
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("rebuild_from_best")


def _build_approach(
    best: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> ForecastingApproach:
    """Build a ForecastingApproach from 'best'."""
    logger = _get_logger(logger)
    conf = _conf_from_best(best)
    args = best.get("strategy_args", {})
    name, atype = conf["name"], conf.get("approach")

    if atype == "hybrid" and "group" in conf:
        conf = {**conf, "data_module_config": args.get("data_module_config", {})}
        app = HybridApproach(config=conf)
        app.config, app.name = conf, name
        return app

    return build_approaches({"model": {name: conf}}, args, logger)[name]


def build_approach_and_model(
    best: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Tuple[ForecastingApproach, Optional[object]]:
    """Build an approach and also initialize model(s) if applicable."""
    app = _build_approach(best, logger=logger)
    if isinstance(app, (DefaultApproach, UnivariateToMultivariate)):
        model: Optional[object] = None
        factory: Optional[Callable[[], object]] = getattr(app, "model_factory", None)
        if factory is not None:
            model = factory()
            setattr(app, "model", model)
        return app, model
    if isinstance(app, HybridApproach):
        conf: Dict[str, Any] = getattr(app, "config", {}) or {}
        groups = conf.get("group") or []
        group_models: Dict[str, Any] = {}
        for grp in groups:
            cols = grp.get("series") or []
            key = ",".join(cols) if cols else grp.get("name") or "group"
            model_name = grp["model_name"]
            model_hparams = grp.get("model_hyper_params", {})
            base_model = get_single_model(model_name, model_hparams)
            group_models[key] = base_model
        setattr(app, "initial_group_models", group_models)
        return app, group_models

    return app, None
