# -*- coding: utf-8 -*-
import os
import gc
import glob
import logging
from pathlib import Path
import traceback
from typing import Union
import warnings

from clearml import Task
from parallelbar import progress_map
import yaml

from common.constants import GENERAL_EXTENSION
from common.enums import PipelineSteps
from preprocess.loaders import PickleLoader
from features.feature_engineer import FeatureEngineer
from settings import Settings
from utilities.utils import check_directory, compress_pickle, get_abs_path

warnings.simplefilter(action="ignore", category=FutureWarning)

settings = Settings()


def make_features(file_path: Union[Path, str]) -> Union[str, None]: 
    
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
         
    file_name = Path(file_path).stem
    features_dataframe_filepath = Path(os.path.join(output_filepath, f"{file_name}{GENERAL_EXTENSION}"))

    if features_dataframe_filepath.is_file():
        task.logger.report_text(f"{file_name} is already processed", level=logging.INFO)
        return file_name
    
    data = PickleLoader(path=file_path).load()
    
    try:    
        fe = FeatureEngineer()
        features = fe.transform(data=data)
    except Exception as exception:
        task.logger.report_text(
            f"Feature engineer for {file_name} failed due to: {exception}", 
            level=logging.INFO)
        task.logger.report_text(
            'traceback:' + traceback.format_exc(), 
            level=logging.DEBUG)
        return exception
    
    compress_pickle(features_dataframe_filepath, features)
    task.logger.report_text(f"Features successfully saved for {file_name}", level=logging.DEBUG)

    del features, data
    gc.collect()
    
    return file_name

if __name__ == "__main__":
    task = Task.init(project_name=settings.clearml.project,
                     task_name=f'{PipelineSteps.feature_engineer} task',
                     task_type=Task.TaskTypes.data_processing,
                     tags=settings.clearml.tags,
                     reuse_last_task_id=False)
    if settings.clearml.execute_remotely:
        task.execute_remotely(queue_name=settings.clearml.queue_name)

    input_filepath = get_abs_path(settings.storage.pipeline_processed_folder)
    file_type = f"/*{GENERAL_EXTENSION}"
    input_filepath_files = glob.glob(str(input_filepath) + file_type)
    output_filepath = check_directory(get_abs_path(settings.storage.pipeline_features_folder))
    
    with open(settings.params_path) as file:
        params = yaml.load(file, Loader=yaml.Loader)
        common_params = params.get('common')
        step_params = params.get(PipelineSteps.feature_engineer)
    if common_params:
        task.connect(common_params, name="common")  
    if step_params:
        task.connect(step_params, name=PipelineSteps.feature_engineer.replace('_', ' '))
    
    result = progress_map(make_features, input_filepath_files, 
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
        f"Feature engineering is finished, data was stored at {output_filepath}",
        level=logging.INFO)