# -*- coding: utf-8 -*-
"""Module with storage utils"""
import logging
from os import listdir
from os.path import exists, isfile, join, getctime
from pathlib import Path
import shutil
from typing import List, Union, Optional

LOGGER = logging.getLogger(__name__)


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



def delete_directory(path: Union[str, Path]) -> None:
    """Removes directory and its files.

    Args:
        path (Union[str, Path]): Directory which should be removed.
    """

    try:
        shutil.rmtree(path)
    except OSError as e:
        LOGGER.warning(f"{e.strerror}: {e.filename}")


def get_last_modified(path: Union[str, Path], suffixes: Optional[List] = None) -> Path:
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
    return max(read_files, key=getctime)


def is_empty_dir(path: Union[str, Path]) -> bool:
    """Checking if the directory is empty or not.

    Args:
        path (Union[str, Path]): checked path.

    Raises:
        ValueError: in case of missing path or file path.

    Returns:
        bool: True or False.
    """    
    if exists(path) and not isfile(path):
  
        # Checking if the directory is empty or not
        if not listdir(path):
            return True
        else:
            return False
    else:
        raise ValueError("The path is either for a file or not valid")
    
    
def get_directory_files(path: Union[str, Path]) -> List[str]:
    return [f for f in listdir(path) if isfile(join(path, f))]