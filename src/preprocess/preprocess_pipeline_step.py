# -*- coding: utf-8 -*-
from typing import Optional, TYPE_CHECKING
import warnings

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from common.pipeline_steps import PREPROCESS
from core import BasePipelineStep
from preprocess.preprocessor import Preprocessor, MarkDataTransformer

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep
    from settings import Settings

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step: 'PipelineStep' = PREPROCESS
        super().__init__(settings, self.pipeline_step)

    def start(
        self,
        data: pd.DataFrame,
        target: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:

        # Configure pipeline
        if not target.empty or target:
            step_pipeline = Pipeline(
                steps=[
                    ("preprocessor", Preprocessor()),
                    ("add_target", MarkDataTransformer(target=target)),
                ]
            )
        else:
            step_pipeline = Pipeline(
                [
                    ("preprocessor", Preprocessor())
                ]
            )

        set_config(transform_output="pandas")
        # Transform data
        try:
            preprocessed = step_pipeline.transform(data)
            self._log_success_step_execution()
        except Exception as exception:
            self._log_failed_step_execution(exception=exception)
            raise exception

        return preprocessed
