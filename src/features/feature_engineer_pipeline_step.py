# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING
import warnings

import pandas as pd

from common.pipeline_steps import FEATURE_ENGINEER
from core import BasePipelineStep
from features.feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep
    from settings import Settings

warnings.simplefilter(action="ignore", category=FutureWarning)


class FeatureEngineerPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step: 'PipelineStep' = FEATURE_ENGINEER
        super().__init__(settings, self.pipeline_step)

    def start(self, data: pd.DataFrame) -> pd.DataFrame:

        fe = FeatureEngineer()
        # Transform data
        try:
            features = fe.transform(data)
            self._log_success_step_execution()
        except Exception as exception:
            self._log_failed_step_execution(exception=exception)
            return exception

        return features
