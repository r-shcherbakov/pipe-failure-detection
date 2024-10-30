# -*- coding: utf-8 -*-
from typing import Optional, TYPE_CHECKING
import warnings

import pandas as pd
from sklearn import set_config
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from common.features import GROUP_ID
from common.pipeline_steps import PREPROCESS
from core import BasePipelineStep
from preprocess.preprocessor import Preprocessor, MarkDataTransformer

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessPipelineStep(BasePipelineStep):
    def __init__(self):
        self.pipeline_step: 'PipelineStep' = PREPROCESS
        super().__init__(self.pipeline_step)

    def _sample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        sample_size = self.step_params.get('sample_size', 1)
        # validate sample size
        sample_size = max(0, min(sample_size, 1))
        if sample_size == 1:
            sample_inds = data.index
        else:
            splitter = GroupShuffleSplit(
                train_size=sample_size,
                n_splits=1,
                random_state=self.settings.random_seed)
            split = splitter.split(data, groups=data[GROUP_ID.name])
            sample_inds, _ = next(split)

        return data.iloc[sample_inds].reset_index(drop=True)

    def start(
        self,
        data: pd.DataFrame,
        target: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:

        # Shrink data to speed up the locally testing of the pipeline
        # Be carefull with it because it affects the distribution of classes
        data = self._sample_data(data)
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
