# -*- coding: utf-8 -*-
"""Module defines enums."""
from enum import Enum


class DefectType(int, Enum):
    """
    Enum for kick type.
    """

    OD = 0
    ID = 1
    UNDEFINED = -1
