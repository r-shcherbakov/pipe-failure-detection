# -*- coding: utf-8 -*-
r"""
Preprocessor of the main input data
"""
import logging

import pandas as pd

from core import BaseLoader

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