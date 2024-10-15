# -*- coding: utf-8 -*-
"""Module describes main features in the project."""
from dataclasses import dataclass
from typing import Optional, Union
from common.constants import NULL, SECONDS_IN_MINUTE


@dataclass(frozen=True)
class Feature:
    """Dataclass for describing features"""

    name: str
    dtype: str
    description: Optional[str] = None
    lower: Optional[Union[int, float]] = None
    upper: Optional[Union[int, float]] = None
    fillna_value: Optional[Union[int, float]] = None
    fillna_method: Optional[str] = "ffill"
    fillna_limit: Optional[int] = None

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __contains__(self, item: Union[str, "Feature", None]):
        if item is None:
            return False
        else:
            return str(item) in self.name


DATETIME = Feature(
    name="time",
    description="measurement time, sec.",
    dtype="timedelta64[s]",
    fillna_limit=SECONDS_IN_MINUTE * 5,
)
TARGET = Feature(
    name="target",
    description="external (OD) or internal (ID) type of defect",
    dtype="category",
    fillna_value=NULL,
    fillna_limit=SECONDS_IN_MINUTE * 5,
)
GROUP_ID = Feature(
    name="id",
    description="measurement identificator",
    dtype="category",
)
CHANNEL_1 = Feature(
    name="ch0",
    description="recordings from three channels over pipe length",
    dtype="float32",
    lower=0,
    upper=1000,
    fillna_value=0,
    fillna_limit=SECONDS_IN_MINUTE * 5,
)
CHANNEL_2 = Feature(
    name="ch1",
    description="recordings from three channels over pipe length",
    dtype="float32",
    lower=0,
    upper=1000,
    fillna_value=0,
    fillna_limit=SECONDS_IN_MINUTE * 5,
)
CHANNEL_3 = Feature(
    name="ch2",
    description="recordings from three channels over pipe length",
    dtype="float32",
    lower=0,
    upper=1000,
    fillna_value=0,
    fillna_limit=SECONDS_IN_MINUTE * 5,
)


IGNORED_FEATURES = [GROUP_ID]
MANDATORY_FEATURES = [
    DATETIME,
    TARGET,
    CHANNEL_1,
    CHANNEL_2,
    CHANNEL_3,
]
