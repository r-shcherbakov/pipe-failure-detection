# -*- coding: utf-8 -*-
import gc
import glob
import logging
import os
from pathlib import Path
import traceback
from typing import Union
import warnings

from clearml import Task
from parallelbar import progress_map
import yaml

from common.constants import GENERAL_EXTENSION
from common.enums import PipelineSteps
from preprocess.preprocessor import Preprocessor
from preprocess.loaders import CsvLoader
from settings import Settings
from utilities.utils import check_directory, compress_pickle, get_abs_path

warnings.simplefilter(action="ignore", category=FutureWarning)

settings = Settings()


def preprocess_data(file_path: Union[Path, str]) -> Union[str, None]:
    
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    file_name = Path(file_path).stem.replace(" ", "").upper()
    task.logger.report_text(f"Processing of {file_name}", level=logging.DEBUG)
    preprocessed_filepath = Path(os.path.join(output_filepath, f"{file_name}{GENERAL_EXTENSION}"))
    
    if preprocessed_filepath.is_file():
        task.logger.report_text(f"{file_name} is already processed", level=logging.INFO)
        return file_name
    
    loader = CsvLoader(path=file_path).load()
    data = loader.load()
    
    try:    
        preprocessor = Preprocessor()
        preprocessed = preprocessor.transform(data=data)
    except Exception as exception:
        task.logger.report_text(
            f"Preprocessor for {file_name} failed due to: {exception}", 
            level=logging.INFO)
        task.logger.report_text(
            'traceback:' + traceback.format_exc(), 
            level=logging.DEBUG)
        return exception
    
    compress_pickle(preprocessed_filepath, preprocessed)
    task.logger.report_text(f"Processed data successfully saved for {file_name}", level=logging.DEBUG)
        
    del preprocessed, data
    gc.collect()
    
    return file_name


if __name__ == "__main__":
    task = Task.init(project_name=settings.clearml.project,
                     task_name=f'{PipelineSteps.preprocess} task',
                     task_type=Task.TaskTypes.data_processing,
                     tags=settings.clearml.tags,
                     reuse_last_task_id=False)
    if settings.clearml.execute_remotely:
        task.execute_remotely(queue_name=settings.clearml.queue_name)
        
    with open(settings.params_path) as file:
        params = yaml.load(file, Loader=yaml.Loader)
        common_params = params.get('common')
        step_params = params.get(PipelineSteps.preprocess)
    if common_params:
        task.connect(common_params, name="common")  
    if step_params:
        task.connect(step_params, name=PipelineSteps.preprocess.replace('_', ' '))
    
    input_filepath = get_abs_path(settings.storage.pipeline_raw_folder)
    file_type = r"/*csv"
    input_filepath_files = glob.glob(str(input_filepath) + file_type)            
    output_filepath = check_directory(get_abs_path(settings.storage.pipeline_processed_folder))

    result = progress_map(preprocess_data, input_filepath_files, 
                          n_cpu=settings.parallelbar.n_cpu, 
                          error_behavior=settings.parallelbar.error_behavior,
                          process_timeout=settings.parallelbar.process_timeout)
    processed_holes = [value for value in result if isinstance(value, str)]
    holes_errors = [value for value in result if isinstance(value, Exception)]
    
    task.upload_artifact(
        name='processed_holes', 
        artifact_object={"processed_holes": processed_holes})
    task.upload_artifact(
        name='holes_errors', 
        artifact_object={"holes_errors": holes_errors})
    task.logger.report_text(
        f"Processing of raw data is finished, data was stored at {output_filepath}",
        level=logging.INFO)   
