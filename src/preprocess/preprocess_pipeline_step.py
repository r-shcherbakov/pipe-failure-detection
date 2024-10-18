# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional
import warnings

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from common.pipeline_steps import PipelineStep, PREPROCESS
from core import BasePipelineStep
from preprocess.preprocessor import Preprocessor, MarkDataTransformer
from settings import Settings
from utilities.loaders import CsvLoader

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step: PipelineStep = PREPROCESS
        super().__init__(settings, self.pipeline_step)

    def start(
        self,
        data: pd.DataFrame,
        target: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:

        # Configure pipeline
        if target or not target.empty:
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


def run_preprocess_step(settings: Settings) -> pd.DataFrame:
    data_path = Path(os.path.join(settings.storage.raw_folder, "data.csv"))
    target_path = Path(os.path.join(settings.storage.raw_folder, "target_train.csv"))
    data = CsvLoader(path=data_path).load()
    target = CsvLoader(path=target_path).load()

    preprocessor = PreprocessPipelineStep(settings=settings)
    return preprocessor.start(data=data, target=target)


if __name__ == "__main__":
    run_preprocess_step(settings=Settings())
