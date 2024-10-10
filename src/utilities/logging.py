import logging.handlers
import sys
from logging import Handler
from typing import List, Any, cast

LOGGER: logging.Logger = cast(Any, None)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(thread)s - %(funcName)s - %(lineno)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def set_logging(log_level: int = logging.INFO) -> None:
    """
    setup logging configuration

    :param settings: logging settings
    :return: None
    """

    global LOGGER

    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(log_level)

    log_format: logging.Formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    log_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    log_handler.setLevel(log_level)
    log_handler.setFormatter(log_format)

    handlers: List[Handler] = [log_handler]
    logging.basicConfig(level=log_level, handlers=handlers)