# -*- coding: utf-8 -*-
r"""Preprocessor transformers"""
import logging

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from core import BaseTransformer
from utilities.transformers import (
    DuplicatedColumnsTransformer,
    ColumnsTypeTransformer, 
    ClipTransformer, 
    InfValuesTransformer,
    FillNanTransformer,
    TimeResampler,
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
                ("drop_duplicate_columns", DuplicatedColumnsTransformer()),
                ("convert_columns_type", ColumnsTypeTransformer()),
                ("drop_outliers", ClipTransformer()),
                ("drop_inf_values", InfValuesTransformer()),
                ("fill_nan", FillNanTransformer()),
                ("resampler", TimeResampler()),
            ]
        )
        set_config(transform_output="pandas")
        
        data = common_pipeline.transform(data)
        return data


class MarkDataTransformer(BaseTransformer):
    r"""Transformer for marking preprocessed data according to expert config."""
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Marks data according to label_config.

        Args:
            X (pd.DataFrame): Input dataframe of preprocessed data.

        Returns:
            pd.DataFrame: Input dataframe with labels of event.
        """
        X = self._mark(X)
        return X

    def _mark(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns input dataframe with manual labeling markers as target.

        Args:
            data (pd.DataFrame): Input dataframe of preprocessed data.

        Returns:
            pd.DataFrame: Input dataframe with manual labeling markers as target.

        """
        mask = self._get_mask(data)
        data['TARGET'] = 0
        data.loc[mask, 'TARGET'] = 1
        data['TARGET'] = data['TARGET'].astype("int16")
        return data

    def _get_mask(self, data: pd.DataFrame) -> pd.Series:
        """Returns mask of labels according to manual labeling config.

        Args:
            data (pd.DataFrame): Input dataframe of preprocessed data.

        Returns:
            pd.Series: Mask of labels according to manual labeling config.
        """
        
        # TODO: Set here your mask of labels
        mask = pd.Series(0, index=data.index)
        return mask