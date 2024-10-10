# -*- coding: utf-8 -*-
import gc
import logging
from pathlib import Path
import random
from typing import List, Optional
import warnings

from catboost import (
    Pool,
    CatBoostClassifier,
)
from clearml import Task
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import yaml

from common.constants import GENERAL_EXTENSION, IGNORED_FEATURES
from common.enums import PipelineSteps
from preprocess.loaders import PickleLoader
from settings import Settings
from utilities.transformers import ColumnsTypeTransformer
from utilities.utils import get_last_modified, compress_pickle, get_abs_path

warnings.simplefilter(action="ignore", category=FutureWarning)

settings = Settings()


def store_cv_results(
    data: pd.DataFrame,
) -> None:
    """Returns the results of common metrics
        for each fold provided by GroupKFold cross-validation.

    Args:
        data (pd.DataFrame): Training dataframe.
    """

    X = data.drop(axis=1, columns="TARGET").copy()
    y = data["TARGET"].fillna(0).copy()
    groups = data["GROUP_ID"].copy()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    n_splits = step_params.pop("n_splits", 2)

    skf = GroupKFold(n_splits=n_splits)
    cv_result = pd.DataFrame()
    for train_idx, valid_idx in tqdm(skf.split(X, y, groups=groups), total=n_splits):
        task.logger.report_text(
            f"Train GROUP_ID:{X.loc[train_idx, 'GROUP_ID'].unique()}",
            level=logging.DEBUG
        )
        task.logger.report_text(
            f"Test GROUP_ID:{X.loc[valid_idx, 'GROUP_ID'].unique()}",
            level=logging.DEBUG
        )

        train_pool = Pool(
            data=X.loc[train_idx],
            label=y.loc[train_idx],
            group_id=groups[train_idx],
            cat_features=cat_features,
        )
        eval_pool = Pool(
            data=X.loc[valid_idx],
            label=y.loc[valid_idx],
            group_id=groups[valid_idx],
            cat_features=cat_features,
        )

        cb_model = CatBoostClassifier(
            **step_params,
            random_seed=settings.random_seed,
        ).fit(
            train_pool,
            eval_set=eval_pool,
        )

        fold_result = {}
        metrics = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
        for metric in metrics:
            fold_result[f"{metric}"] = cb_model.eval_metrics(eval_pool, [metric])[
                metric
            ][-1]
        fold_result = pd.DataFrame([fold_result])
        cv_result = pd.concat([cv_result, fold_result])
        del train_pool, eval_pool
        gc.collect()
        
    cv_result.loc["mean"] = cv_result.mean()
    task.logger.report_table(
        title="CV results", 
        series="CV results",
        table_plot=cv_result
    ) 


if __name__ == "__main__":
    task = Task.init(project_name=settings.clearml.project,
                    task_name=f'{PipelineSteps.train} task',
                    task_type=Task.TaskTypes.training,
                    tags=settings.clearml.tags,
                    reuse_last_task_id=False)
    if settings.clearml.execute_remotely:
        task.execute_remotely(queue_name=settings.clearml.queue_name)
    
    # Get step params
    with open(settings.params_path) as file:
        params = yaml.load(file, Loader=yaml.Loader)
        common_params = params.get('common')
        step_params = params.get(PipelineSteps.train)
    if common_params:
        task.connect(common_params, name="common")  
    if step_params:
        task.connect(step_params, name=PipelineSteps.train.replace('_', ' '))
        
    input_filepath = get_last_modified(
        get_abs_path(settings.storage.pipeline_train_folder), GENERAL_EXTENSION
    )
    data = PickleLoader(path=input_filepath).load()
    ignored_features = list(np.intersect1d(data.columns.tolist(), IGNORED_FEATURES))
    step_params["ignored_features"] = ignored_features
    
    store_cv_results(data)
        
        
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