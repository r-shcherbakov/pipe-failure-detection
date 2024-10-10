"""pipe_failure_detection."""

import warnings

from loguru import logger

from importlib.metadata import version

__version__ = version(__package__)

__version__ = '0.0.1'
__pkg_name__ = 'pipe_failure_detection'

logger.disable(__pkg_name__)
