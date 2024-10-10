# -*- coding: utf-8 -*-
r"""
Preprocessor of the main input data
"""
import logging
from abc import ABCMeta
from pathlib import Path, PosixPath
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from common.exceptions import FileTypeError
from core.loader import BaseLoader

LOGGER = logging.getLogger(__name__)


class CsvLoader(BaseLoader):       
    r"""Loads raw csv data files."""

    def load(self) -> pd.DataFrame:
        data = pd.read_csv(self.path, engine="pyarrow")
        return data
    
    
class PickleLoader(BaseLoader):       
    r"""Loads raw pickle data files."""

    def load(self) -> pd.DataFrame:
        data = pd.read_pickle(self.path, compression={"method": "gzip"})
        return data