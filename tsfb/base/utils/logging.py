import logging
import sys
from datetime import datetime
from typing import Any, Callable, List, Mapping, MutableMapping, Tuple, Union, cast

import structlog
from colorama import Fore, Style, init

init(autoreset=True)


def _color(log: str) -> str:
    return {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }.get(log.upper(), Fore.WHITE)


def _exc(_l: Any, _n: str, d: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if exc := d.pop("exc_info", None):
        t, v = exc[:2] if isinstance(exc, tuple) else (type(exc), exc)
        d["exception"] = f"{t.__name__}: {v}"
    return d


def _render(_: Any, n: str, d: MutableMapping[str, Any]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lvl, msg, mod = (
        d.pop("level", "INFO").upper(),
        d.pop("event", ""),
        d.pop("module", n or "-"),
    )
    if exc := d.pop("exception", None):
        msg += f" | {exc}"
    ext = " ".join(f"{k}={v}" for k, v in d.items())
    return (
        f"{Fore.CYAN}{ts}{Style.RESET_ALL} - {Fore.MAGENTA}{mod}{Style.RESET_ALL} - "
        f"{_color(lvl)}[{lvl}]{Style.RESET_ALL} - {Style.BRIGHT}{msg}{Style.RESET_ALL}"
        + (f" {Fore.CYAN}{ext}{Style.RESET_ALL}" if ext else "")
    )


def setup_logging(lvl: str = "INFO", trace: bool = False) -> None:
    """Setup structured colorized logging."""
    lv = getattr(logging, lvl.upper(), logging.INFO)
    logging.basicConfig(level=lv, format="%(message)s", stream=sys.stdout)
    P = Callable[
        [Any, str, MutableMapping[str, Any]],
        Union[Mapping[str, Any], str, bytes, bytearray, Tuple[Any, ...]],
    ]
    ps: List[P] = [
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        cast(P, structlog.processors.format_exc_info if trace else _exc),
        cast(P, _render),
    ]
    structlog.configure(
        processors=ps,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(lv),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Return configured structlog logger."""
    if not structlog.is_configured():
        setup_logging()
    return structlog.get_logger(name).bind(module=name)
