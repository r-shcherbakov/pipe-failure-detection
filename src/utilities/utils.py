# -*- coding: utf-8 -*-
"""Main project utils"""
import os
import logging
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from common.config import ALL_TYPES
from common.constants import GENERAL_EXTENSION
from settings import PROJECT_PATH

LOGGER = logging.getLogger(__name__)


def get_abs_path(path: Union[str, Path]) -> Path:
    """Returns absolute path.

    Args:
        path (Union[str, Path]): Relative path.

    Returns:
        Path: Absolute path.
    """

    if str(PROJECT_PATH) in str(path):
        return Path(path)
    else:
        return Path(os.path.join(PROJECT_PATH, path))
    
    
def get_last_modified(path: Union[str, Path], suffixes: list = None) -> Path:
    """Returns path of last modified file with required suffix.

    Args:
        path (Union[str, Path]): Input directory.
        suffixes (list, optional): List of required suffixes.
            Defaults to None.

    Returns:
        Path: Path of last modified file with required suffix.
    """

    path = Path(path)
    path_files = path.iterdir()
    suffixes = suffixes if suffixes is not None else [""]
    read_files = filter(
        lambda path_files: path_files.suffix in suffixes, Path(path).rglob("*.*")
    )
    return max(read_files, key=os.path.getctime)
    
    
def check_directory(filepath: Union[str, Path]) -> Path:
    """Check and create missing directory.

    Args:
        filepath (Union[str, Path]): File path of required directory.

    Returns:
        Path: File path of required directory.
    """

    filepath = Path(filepath)
    filepath.mkdir(exist_ok=True, parents=True)
    return filepath
    

def compress_pickle(path: Path, data: pd.DataFrame) -> None:
    """Saves compressed dataframe to storage.

    Args:
        path (Path): Path to output directory.
        data (pd.DataFrame): Input data.
    """
    file_extension = Path(path).suffix
    if file_extension == '' or file_extension != GENERAL_EXTENSION:
        output_filepath = Path(
            os.path.join(path.parent, (path.name + GENERAL_EXTENSION))
        ) 
    else:
        output_filepath = Path(path)
    data.to_pickle(
        output_filepath, compression={"method": "gzip", "compresslevel": 1, "mtime": 1}
    )


def convert_columns_type(data: pd.DataFrame) -> pd.DataFrame:
    """Converts column values type according to config.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Input data with converted columns types.
    """

    columns = data.columns.tolist()
    types_dict = {key: ALL_TYPES[key] for key in columns & ALL_TYPES.keys()}
    undefined_columns = np.setdiff1d(np.unique(columns), ALL_TYPES.keys())
    if len(undefined_columns) > 0:
        reduce_memory_usage(data[undefined_columns])
    return data.astype(types_dict)


def reduce_memory_usage(data: pd.DataFrame) -> pd.DataFrame:
    """Iterates through all columns of input dataframe and
        converting the data type in order to reduce memory usage.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Input data with converted columns types.
    """

    initial_memory = data.memory_usage().sum() / 1024**2
    LOGGER.debug(f"Initial memory usage of dataframe is {initial_memory:.2f} MB")
    
    for column, dtype in zip(data.columns, data.dtypes):
        dtype = str(dtype)
        column_data = data[column]
        if any([_ in dtype for _ in ("int", "float")]):
            col_min, col_max = column_data.min(), column_data.max()
            if "int" in dtype:
                if (col_min > np.iinfo(np.int8).min) and (col_max < np.iinfo(np.int8).max):
                    column_data = column_data.astype(np.int8)
                elif (col_min > np.iinfo(np.int16).min) and (col_max < np.iinfo(np.int16).max):
                    column_data = column_data.astype(np.int16)
                elif (col_min > np.iinfo(np.int32).min) and (col_max < np.iinfo(np.int32).max):
                    column_data = column_data.astype(np.int32)
                elif (col_min > np.iinfo(np.int64).min) and (col_max < np.iinfo(np.int64).max):
                    column_data = column_data.astype(np.int64)
            else:
                if (col_min > np.finfo(np.float16).min) and (col_max < np.finfo(np.float16).max):
                    column_data = column_data.astype(np.float16)
                elif (col_min > np.finfo(np.float32).min) and (col_max < np.finfo(np.float32).max):
                    column_data = column_data.astype(np.float32)
                else:
                    column_data = column_data.astype(np.float64)
        elif ("object" in dtype) or ("category" in dtype):
            column_data = pd.Categorical(column_data)
        elif "datetime" in dtype:
            column_data = column_data.astype("datetime64[ns]")
            
        data[column] = column_data

    final_memory = data.memory_usage().sum() / 1024**2
    LOGGER.debug(f"Memory usage after optimization is {final_memory:.2f} MB")

    return data


def get_common_timestep(data: pd.DataFrame) -> int:
    """Returns the common time step in seconds of input timeseries dataframe.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        int: Common time step in seconds.
    """

    datetime_columns = data.select_dtypes(include=['datetime64']).columns
    common_timestep = None
    for column in datetime_columns:
        timestep = data[column].diff().dt.total_seconds().mode().astype(int)[0]
        common_timestep = timestep if common_timestep is None else min(timestep, common_timestep)
    return common_timestep


def get_subclasses(cls):
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(get_subclasses(subclass))
    return subclasses