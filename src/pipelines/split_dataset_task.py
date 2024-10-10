# -*- coding: utf-8 -*-
import gc
import glob
import logging
from pathlib import Path
import random
from typing import List, Optional
import warnings

from clearml import Task
import pandas as pd
from tqdm import tqdm
import yaml

from common.constants import GENERAL_EXTENSION
from common.enums import PipelineSteps
from preprocess.loaders import PickleLoader
from settings import Settings
from utilities.transformers import ColumnsTypeTransformer
from utilities.utils import check_directory, compress_pickle, get_abs_path

warnings.simplefilter(action="ignore", category=FutureWarning)

settings = Settings()


if __name__ == "__main__":
    task = Task.init(project_name=settings.clearml.project,
                    task_name=f'{PipelineSteps.split_dataset} task',
                    task_type=Task.TaskTypes.data_processing,
                    tags=settings.clearml.tags,
                    reuse_last_task_id=False)
    if settings.clearml.execute_remotely:
        task.execute_remotely(queue_name=settings.clearml.queue_name)
    
    input_filepath = get_abs_path(settings.storage.pipeline_features_folder)
    file_type = f"/*{GENERAL_EXTENSION}"
    input_filepath_files = glob.glob(str(input_filepath) + file_type)
    output_filepath = check_directory(get_abs_path(settings.storage.pipeline_splitted_folder))
    train_filepath = check_directory(get_abs_path(settings.storage.pipeline_train_folder))
    test_filepath = check_directory(get_abs_path(settings.storage.pipeline_test_folder))
    
    # Get step params
    with open(settings.params_path) as file:
        params = yaml.load(file, Loader=yaml.Loader)
        common_params = params.get('common')
        step_params = params.get(PipelineSteps.split_dataset)
    if common_params:
        task.connect(common_params, name="common")  
    if step_params:
        task.connect(step_params, name=PipelineSteps.split_dataset.replace('_', ' '))
    
    add_test = step_params.get("split_test", False)
    if add_test:
        test_objects: Optional[List[str]] = step_params.get("test_objects", None)
        if test_objects is None:
            random.seed(settings.random_seed)
            num_test_objects = step_params.get("num_test_objects", 1)
            test_objects = [
                Path(file_path).stem for file_path in random.sample(input_filepath_files, num_test_objects)
            ]
            step_params["test_objects"] = test_objects
        else:
            step_params["test_objects"] = test_objects
    else:
        test_objects = []
        
    file_name_mapping = {
        Path(file_path).stem: number for number, file_path in enumerate(input_filepath_files)
    }
    task.connect(file_name_mapping, name="file_name_mapping")

    train = pd.DataFrame()
    test = pd.DataFrame()

    for file_path in tqdm(input_filepath_files, total=len(input_filepath_files)):
        file_name = Path(file_path).stem
        task.logger.report_text(f"Processing of {file_name}", level=logging.DEBUG)
        loader = PickleLoader(path=file_path).load()
        data = loader.load()
        data['GROUP_ID'] = file_name_mapping[file_name]
        
        # It's required to reduce memory size of dataframe
        data = ColumnsTypeTransformer().transform(data)

        if file_name in test_objects:
            test = pd.concat([test, data])
        else:
            train = pd.concat([train, data])

        del data
        gc.collect()
        
    compress_pickle(train_filepath, train)
    if add_test:
        compress_pickle(test_filepath, test)

    task.logger.report_text(
        f"Splitting of dataset is finished, data was stored at {output_filepath}",
        level=logging.INFO
    )   
    