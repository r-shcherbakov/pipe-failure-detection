# -*- coding: utf-8 -*-
r"""Preprocessor transformers"""
import logging

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from core import BaseTransformer
from common.features import TARGET, GROUP_ID, DATETIME
from common.config import (
    FEATYPE_TYPES,
    FILLNA_CONFIG,
    CLIP_CONFIG,
)
from common.enums import DefectType
from utilities.transformers import (
    ColumnsTypeTransformer,
    ClipTransformer,
    InfValuesTransformer,
    FillNanTransformer,
)

LOGGER = logging.getLogger(__name__)


class Preprocessor(BaseTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw data with basic preprocess methods and
            predefined or custom pipelines.
        Args:
            data (pd.DataFrame): Input raw data.

        Returns:
            pd.DataFrame: Dataframe of preprocessed data.
        """

        data = X.copy()
        common_pipeline = Pipeline(
            [
                ("drop_outliers", ClipTransformer(config=CLIP_CONFIG)),
                ("drop_inf_values", InfValuesTransformer()),
                ("fill_nan", FillNanTransformer(config=FILLNA_CONFIG)),
                ("convert_columns_type", ColumnsTypeTransformer(config=FEATYPE_TYPES)),
            ]
        )
        set_config(transform_output="pandas")
        data = common_pipeline.transform(data)
        data = data.sort_values(by=[GROUP_ID.name, DATETIME.name]) \
            .reset_index(drop=True)

        return data


class MarkDataTransformer(BaseTransformer):
    r"""Transformer for marking preprocessed data according to expert config."""
    def __init__(self, target: pd.DataFrame):
        self.target = target

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Marks data according to label_config.

        Args:
            X (pd.DataFrame): Input dataframe of preprocessed data.

        Returns:
            pd.DataFrame: Input dataframe with labels of event.
        """
        target_encoding = dict(self.target.values)
        X[TARGET.name] = X[GROUP_ID.name].map(target_encoding) \
            .map({i.name: i.value for i in DefectType}) \
            .fillna(DefectType.UNDEFINED.value) \
            .astype(int)

        return X
