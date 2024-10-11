# -*- coding: utf-8 -*-
"""Exceptions for project."""


class BaseException(Exception):
    """Base class for all exceptions."""


class FileTypeError(BaseException):
    """Raised when loader get the files with wrong suffix."""
    
    
class MarkingDataError(BaseException):
    """Raised when loader aren't able to mark the data."""
    
    
class PipelineExecutionError(BaseException):
    """Raised when pipeline failed during execution."""
    
    
class DatasetDownloadError(BaseException):
    """Raised when pipeline failed during execution."""