# -*- coding: utf-8 -*-
"""Exceptions for project."""


class BaseException(Exception):
    """Base class for all exceptions."""


class FileTypeError(BaseException):
    """Raised when loader get the files with wrong suffix."""


class SplitDataError(BaseException):
    """Raised when data couldn't be splitted to train and test with same destribution."""


class PipelineExecutionError(BaseException):
    """Raised when pipeline failed during execution."""


class DatasetDownloadError(BaseException):
    """Raised when pipeline failed during execution."""
