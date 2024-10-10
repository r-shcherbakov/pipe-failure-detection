# -*- coding: utf-8 -*-
"""Module describes main constants in the project."""
from datetime import timedelta

SECONDS_IN_MINUTE = int(timedelta(minutes=1).total_seconds())
SECONDS_IN_HOUR = int(timedelta(hours=1).total_seconds())

GENERAL_EXTENSION = ".gz"

IGNORED_FEATURES = ["GROUP_ID", "SOME_FORBIDDEN_COLUMN"]
