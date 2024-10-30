# -*- coding: utf-8 -*-
from typing import Tuple, TYPE_CHECKING
import warnings

import pandas as pd

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

        fe = FeatureEngineer()
        try:
            fe.fit(train)
        except Exception as exception:
            self._log_failed_step_execution(exception=exception)
            raise exception

        # Transform data
        try:
            train_features = fe.transform(train)
            test_features = fe.transform(test)
            self._log_success_step_execution()
        except Exception as exception:
            self._log_failed_step_execution(exception=exception)
            raise exception

        return fe, train_features, test_features
