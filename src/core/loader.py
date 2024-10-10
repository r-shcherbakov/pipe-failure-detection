# -*- coding: utf-8 -*-
""" Base loader """
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from typing import Union


class BaseLoader(ABC):
    def __init__(self, 
                 path: Union[str, PosixPath]):
        r"""Loads raw data files.

        Args:
            path (Union[str, PosixPath]): 
                path or several paths to source file.

        Raises:
            ValueError: In case of wrong type of path.
        """        

        if isinstance(path, str) or isinstance(path, PosixPath):
            self.path = Path(path)
        else:
            raise ValueError(f"path should be PosixPath or str. Get {type(path)}")

    @abstractmethod
    def load(self):
        pass