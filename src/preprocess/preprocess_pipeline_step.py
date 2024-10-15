# -*- coding: utf-8 -*-
import os
import gc
from pathlib import Path
from typing import List, TYPE_CHECKING
import warnings

from sklearn import set_config
from sklearn.pipeline import Pipeline

from common.constants import GENERAL_EXTENSION
from common.exceptions import PipelineExecutionError
from common.pipeline_steps import PipelineStep, PREPROCESS
from core import BasePipelineStep
from utilities.loaders import CsvLoader
from preprocess.preprocessor import Preprocessor, MarkDataTransformer

if TYPE_CHECKING:
    from settings import Settings

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step: PipelineStep = PREPROCESS
        super().__init__(settings, self.pipeline_step)

    @property
    def _input_files(self) -> List[Path]:
        pass

    def _upload_artifacts(self) -> None:
        pass

    def _process_data(self) -> None:
        file_name = Path(os.path.join(self._input_directory, "data.csv"))
        data = CsvLoader(path=file_name).load()

        # Configure pipeline
        if self.step_params.get('skip_mark', True):
            step_pipeline = Pipeline(
                [
                    ("preprocessor", Preprocessor())
                ]
            )
        else:
            target = CsvLoader(path=os.path.join(self._input_directory, "target_train.csv")).load()
            step_pipeline = Pipeline(
                steps=[
                    ("preprocessor", Preprocessor()),
                    ("add_target", MarkDataTransformer(target=target)),
                ]
            )
        set_config(transform_output="pandas")

        # Transform data
        try:
            preprocessed = step_pipeline.transform(data)
            self._log_success_step_execution(file_name=file_name)
        except Exception as exception:
            self._log_failed_step_execution(
                file_name=file_name,
                exception=exception,
            )
            return exception

        # Save locally data
        preprocessed_filepath = Path(
            os.path.join(
                self._output_directory, f"{file_name}{GENERAL_EXTENSION}"
            )
        )
        try:
            self._save_locally_data(
                path=preprocessed_filepath,
                data=preprocessed,
            )
        except PipelineExecutionError as exception:
            return exception

        del preprocessed, data
        gc.collect()

        return file_name
