# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
import traceback
from typing import List, Union, TYPE_CHECKING
import warnings

from common.constants import GENERAL_EXTENSION
from common.exceptions import PipelineExecutionError
from common.pipeline_steps import FEATURE_ENGINEER
from core import BasePipelineStep
from features.feature_engineer import FeatureEngineer
from utilities.loaders import PickleLoader

if TYPE_CHECKING:
    from settings import Settings

warnings.simplefilter(action="ignore", category=FutureWarning)


class FeatureEngineerPipelineStep(BasePipelineStep):
    def __init__(
        self,
        settings: 'Settings'
    ):
        self.pipeline_step = FEATURE_ENGINEER
        super().__init__(settings, self.pipeline_step)
        
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
        processed_objects = [value for value in self.result if isinstance(value, str)]
        initial_files = set(
            file_path.stem.replace(" ", "").upper() for file_path in self._input_files
        )
        processing_errors: List[str] = list(initial_files - set(processed_objects))
        
        self.task.upload_artifact(
            name='processed_objects', 
            artifact_object={"processed_objects": processed_objects})
        self.task.upload_artifact(
            name='processing_errors', 
            artifact_object={"processing_errors": processing_errors})
    
    def _process_data(self) -> None:
        train_input_directory = Path(os.path.join(self._input_directory, "train"))
        train = PickleLoader(path=train_input_directory).load()
        try :
            test = PickleLoader(path=Path(os.path.join(self._input_directory, "test"))).load()
        except FileNotFoundError:
            test = None
              
        try:    
            fe = FeatureEngineer()
            fe.fit(train)
        except Exception as exception:
            self.task.logger.report_text(
                f"Featute Engineer fit failed due to: {exception}", 
                level=logging.INFO
            )
            self.task.logger.report_text(
                'traceback:' + traceback.format_exc(), 
                level=logging.DEBUG,
                print_console=False,
            )
            raise PipelineExecutionError

        self.task.upload_artifact(
            name='feature_engineer', 
            artifact_object={"feature_engineer": fe}
        )

        train = fe.transform(train)        
        train_output_directory = Path(os.path.join(
            self._output_directory, 
            f"train{GENERAL_EXTENSION}"
        ))
        self._save_locally_data(
            path=train_output_directory,
            data=train,
        )
        
        if test or not test.empty:
            test = fe.transform(test)
            test_output_directory = Path(os.path.join(
                self._output_directory, 
                f"test{GENERAL_EXTENSION}"
            ))
            self._save_locally_data(
                path=test_output_directory,
                data=test,
            )

        del train, test
        gc.collect()
