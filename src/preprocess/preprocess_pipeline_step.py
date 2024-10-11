# -*- coding: utf-8 -*-
import os
import gc
from glob import glob
import logging
from pathlib import Path
from typing import List, Union, TYPE_CHECKING
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
        self._check_input_directory()
        input_directory = self._input_directory
        file_type = r"/*csv"
        input_filepath_files = [
            Path(file_path) for file_path in glob(str(input_directory) + file_type)
        ]
        return input_filepath_files
    
    def _upload_artifacts(self) -> None:
        processed_objects: List[str] = [value for value in self.result if isinstance(value, str)]
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
    
    def _transform_input_data(self, file_path: Union[Path, str]):
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
    
        file_name = Path(file_path).stem.replace(" ", "").upper()
        self.task.logger.report_text(
            f"Processing of {file_name}", 
            level=logging.DEBUG,
            print_console=False,
        )
        
        data = CsvLoader(path=file_path).load()
        # Configure pipeline
        if self.step_params.get('skip_mark', True):
            step_pipeline = Pipeline(
                [
                    ("preprocessor", Preprocessor())
                 ]
            )
        else:
            step_pipeline = Pipeline(
                steps=[
                    ("preprocessor", Preprocessor()),
                    ("add_target", MarkDataTransformer()),
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
    
    def _process_data(self) -> None:
        self.result = []
        for path in self._input_files:
            self.result.append(self._transform_input_data(path))
