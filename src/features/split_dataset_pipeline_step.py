# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, TYPE_CHECKING
import warnings

import pandas as pd
from tqdm import tqdm

from common.constants import GENERAL_EXTENSION
from common.exceptions import PipelineExecutionError
from common.pipeline_steps import SPLIT_DATASET, PREPROCESS
from core import BasePipelineStep
from utilities.loaders import PickleLoader

if TYPE_CHECKING:
    from settings import Settings

warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitDatasetPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step = SPLIT_DATASET
        self.previous_pipeline_step = PREPROCESS
        super().__init__(
            settings=settings,
            pipeline_step=self.pipeline_step,
            previous_pipeline_step=self.previous_pipeline_step,
        )

    @property
    def _input_files(self) -> List[Path]:
        self._check_input_directory()
        input_directory = self._input_directory
        file_type = f"/*{GENERAL_EXTENSION}"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files

    def _upload_artifacts(self) -> None:
        pass

    def _set_test_objects(self) -> None:
        split_test = self.step_params.get('split_test', False)
        if split_test:
            self.test_objects: Optional[List[str]] = self.step_params.get("test_objects", None)
            if self.test_objects is None:
                # Set required train test split method
                random.seed(self.settings.random_seed)
                num_test_objects = self.step_params.get("num_test_objects", 1)
                self.test_objects = [
                    Path(file_path).stem \
                    for file_path in random.sample(self._input_files, num_test_objects)
                ]
                self.step_params["test_objects"] = self.test_objects
            else:
                self.step_params["test_objects"] = self.test_objects
        else:
            self.test_objects = []

    def _process_data(self) -> None:
        self._set_test_objects()

        train = pd.DataFrame()
        test = pd.DataFrame()
        try:
            for file_path in tqdm(self._input_files, total=len(self._input_files)):
                file_name = Path(file_path).stem
                self.task.logger.report_text(
                    f"Processing of {file_name}",
                    level=logging.DEBUG,
                    print_console=False,
                )
                data = PickleLoader(path=file_path).load()
                data['GROUP_ID'] = self.file_name_mapping[file_name]

                if file_name in self.test_objects:
                    test = pd.concat([test, data])
                else:
                    train = pd.concat([train, data])

                del data
                gc.collect()

        except Exception as exception:
            self._log_failed_step_execution(
                file_name=file_name,
                exception=exception,
            )
            raise PipelineExecutionError

        self._save_locally_data(
            path=Path(os.path.join(self._output_directory, "train")),
            data=train,
        )

        if self.step_params.get('split_test', False) and not test.empty:
            self._save_locally_data(
                path=Path(os.path.join(self._output_directory, "test")),
                data=test,
            )
