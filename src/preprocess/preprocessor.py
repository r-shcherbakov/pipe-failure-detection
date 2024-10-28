# -*- coding: utf-8 -*-
r"""Preprocessor transformers"""
import logging
from typing import List, Optional

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.transformers import FeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute


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
    def __init__(self, selected_fc_features: Optional[List[str]] = None):
        if selected_fc_features:
            # Get config from columns names
            self.selected_fc_features = from_columns(selected_fc_features)
        else:
            self.selected_fc_features = None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw data with basic preprocess methods and
            predefined or custom pipelines.
        Args:
            data (pd.DataFrame): Input raw data.

        Returns:
            pd.DataFrame: Dataframe of preprocessed data.
        """

        data = X.copy()
        data = data.sort_values(by=[GROUP_ID.name, DATETIME.name]) \
            .reset_index(drop=True)

        common_pipeline = Pipeline([
            ("drop_outliers", ClipTransformer(config=CLIP_CONFIG)),
            ("drop_inf_values", InfValuesTransformer()),
            ("fill_nan", FillNanTransformer(config=FILLNA_CONFIG)),
            ('augmenter', FeatureAugmenter(
                column_id=GROUP_ID.name,
                column_sort=DATETIME.name,
                impute_function=impute,
                disable_progressbar=True,
                kind_to_fc_parameters=self.selected_fc_features,
            )),
            ("convert_columns_type", ColumnsTypeTransformer(config=FEATYPE_TYPES)),
        ])
        set_config(transform_output="pandas")
        common_pipeline.set_params(augmenter__timeseries_container=data);

        output = pd.DataFrame(index=data[GROUP_ID.name].unique())
        output = common_pipeline.transform(output)

        return output


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
        data = X.copy()
        target_encoding = dict(self.target.values)
        data[TARGET.name] = data.index.map(target_encoding) \
            .map({i.name: i.value for i in DefectType}) \
            .fillna(DefectType.UNDEFINED.value) \
            .astype(int)

        return data
