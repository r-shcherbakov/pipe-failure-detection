# -*- coding: utf-8 -*-
from typing import Tuple, TYPE_CHECKING
import warnings

import pandas as pd

from common.features import TARGET
from common.pipeline_steps import FEATURE_ENGINEER
from core import BasePipelineStep
from features.feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)


class FeatureEngineerPipelineStep(BasePipelineStep):
    def __init__(self):
        self.pipeline_step: 'PipelineStep' = FEATURE_ENGINEER
        super().__init__(self.pipeline_step)

    def start(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) ->  Tuple['FeatureEngineer', pd.DataFrame, pd.DataFrame]:

        fe = FeatureEngineer(
            random_state=self.settings.random_seed,
            pca_n_components=self.step_params.get('pca_n_components', 3),
            kmeans_n_clusters=self.step_params.get('kmeans_n_clusters', 4),
            kmeans_init=self.step_params.get('kmeans_init', 'k-means++'),
        )
        try:
            fe.fit(train.drop(columns=[TARGET.name]))
        except Exception as exception:
            self._log_failed_step_execution(exception=exception)
            raise exception

        # Transform data
        try:
            train_features = pd.concat(
                [fe.transform(train.drop(columns=[TARGET.name])), train[[TARGET.name]]],
                axis='columns',
            )
            if not test.empty:
                test_features = pd.concat(
                    [fe.transform(test.drop(columns=[TARGET.name])), test[[TARGET.name]]],
                    axis='columns',
                )
            else:
                test_features = pd.DataFrame(columns=train_features.columns)
            self._log_success_step_execution()
        except Exception as exception:
            self._log_failed_step_execution(exception=exception)
            raise exception

        return fe, train_features, test_features
