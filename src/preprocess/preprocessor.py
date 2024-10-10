# -*- coding: utf-8 -*-
r"""Preprocessor transformers"""
import logging
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from common.exceptions import MarkingDataError
from core.transformer import BaseTransformer
from utilities.transformers import DuplicatedColumnsTransformer, ColumnsTypeTransformer, \
    ClipTransformer, InfValuesTransformer, FillNanTransformer, TimeResampler

LOGGER = logging.getLogger(__name__)


class Preprocessor(BaseTransformer):
    def __init__(self, skip_mark: bool = False, extra_pipeline: Optional[Pipeline] = None):
        self.skip_mark = skip_mark
        self.extra_pipeline = extra_pipeline
        if self.extra_pipeline:
            self.extra_pipeline.set_output(transform="pandas")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = RawDataTransformer(extra_pipeline=self.extra_pipeline).transform(data=data)
        if not self.skip_mark:
            result = self._mark_data(result)
        return result
    
    def _mark_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Set labels for each datetime according to manual labeling config.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input dataframe with labels of predicted event.
        """

        try:
            data = MarkDataTransformer().transform(data)
        except MarkingDataError as exception:
            LOGGER.warning(f"Unable to mark data for file {self.path} due to: {exception}")
                
        return data


class RawDataTransformer(BaseTransformer):
    def __init__(self, extra_pipeline: Optional[Pipeline] = None):
        """Preprocesses raw data according to predefined or custom pipeline.

        Args:
            extra_pipeline (Optional[Pipeline], optional): Sequence of additional
                transformations using set of transformers.. Defaults to None.
        """
        
        self.extra_pipeline = extra_pipeline if isinstance(extra_pipeline, Pipeline) else None

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw data with basic preprocess methods and
            predefined or custom pipelines.
        Args:
            data (pd.DataFrame): Input raw data.

        Returns:
            pd.DataFrame: Dataframe of preprocessed data.
        """
        
        df = data.copy()
        common_pipeline = Pipeline(
            [
                ("drop_duplicate_columns", DuplicatedColumnsTransformer()),
                ("convert_columns_type", ColumnsTypeTransformer()),
                ("drop_outliers", ClipTransformer()),
                ("drop_inf_values", InfValuesTransformer()),
                ("fill_nan", FillNanTransformer()),
                ("resampler", TimeResampler()),
            ]
        ).set_output(transform="pandas")
        df = common_pipeline.transform(df)
        if self.extra_pipeline is not None:
            df = self.extra_pipeline.transform(df)
        return df


class MarkDataTransformer(BaseTransformer):
    r"""Transformer for marking preprocessed data according to expert config."""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Marks data according to label_config.

        Args:
            data (pd.DataFrame): Input dataframe of preprocessed data.

        Returns:
            pd.DataFrame: Input dataframe with labels of event.
        """

        data = self._mark(data)
        return data

    def _mark(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns input dataframe with manual labeling markers as target.

        Args:
            data (pd.DataFrame): Input dataframe of preprocessed data.

        Returns:
            pd.DataFrame: Input dataframe with manual labeling markers as target.

        """
        mask = self._get_mask(data)
        data['target'] = 0
        data.loc[mask, 'target'] = 1
        data['target'] = data['target'].astype("int16")
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