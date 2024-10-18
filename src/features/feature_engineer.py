# -*- coding: utf-8 -*-
""" Base feature engineer """
import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline, make_pipeline
from tsfresh.transformers import FeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

from common.features import TARGET, GROUP_ID, DATETIME
from core import BaseTransformer
from utilities.transformers import (
    ColumnsTypeTransformer,
    FillNanTransformer,
)

LOGGER = logging.getLogger(__name__)


class FeatureEngineer(BaseTransformer):

    def fit(self, X, y=None):
        self.prefitted_pipeline = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        y = X.groupby(GROUP_ID.name)[TARGET.name].max()
        time_series = X.drop(axis=1, columns=TARGET.name)
        self.initial_columns = time_series.columns
        X = pd.DataFrame(index=y.index)

        combined_pipeline = Pipeline([
            ("custom", self.custom_pipeline),
            ("postprocessing", self.postprocessing_pipeline),
        ])
        combined_pipeline = make_pipeline(
            self.custom_pipeline,
            self.postprocessing_pipeline,
        )
        set_config(transform_output="pandas")
        combined_pipeline.set_params(custom__augmenter__timeseries_container=time_series);

        features = combined_pipeline.transform(X)
        self.output_columns = X.columns
        output = pd.concat([features, y], axis="columns")

        return output

    @property
    def postprocessing_pipeline(self) -> 'Pipeline':
        pipeline = [
            ("fill_nan", FillNanTransformer()),
            ("convert_columns_type", ColumnsTypeTransformer()),
        ]
        return Pipeline(pipeline)

    @property
    def custom_pipeline(self) -> 'Pipeline':
        return Pipeline(
            ('augmenter', FeatureAugmenter(
                column_id=GROUP_ID.name,
                column_sort=DATETIME.name,
                impute_function=impute,
                disable_progressbar=True,
            )),
        )

    @property
    def new_columns(self) -> List[str]:
        r"""Columns which were generated by feature engineer"""
        new_columns = np.setdiff1d(
            np.unique(self.output_columns),
            self.initial_columns
        )
        return new_columns
