import os
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

from clearml import PipelineController, Dataset
import pandas as pd

from common.exceptions import PipelineExecutionError
from common.features import TARGET
from common.pipeline_steps import (
    PRERUN,
    PREPROCESS,
    FEATURE_ENGINEER,
    SELECT_FEATURES,
    SPLIT_DATASET,
)
from features import (
    FeatureEngineerPipelineStep,
    SplitDatasetPipelineStep,
    SelectFeaturesPipelineStep,
)
from preprocess import PreprocessPipelineStep
from settings import SETTINGS
from utilities.loaders import CsvLoader
from utilities.path_utils import is_empty_dir

if TYPE_CHECKING:
    from features.feature_engineer import FeatureEngineer


def run_prerun_step() -> str:
    if not is_empty_dir(SETTINGS.storage.raw_folder):
        try:
            remote_dataset = Dataset.get(
                dataset_project=SETTINGS.clearml.project,
                dataset_name=f"{SETTINGS.clearml.project} raw data",
                dataset_tags=SETTINGS.clearml.tags,
                only_completed=True,
            )
        except ValueError:
            remote_dataset = None

        local_dataset = Dataset.create(
            dataset_project=SETTINGS.clearml.project,
            dataset_name=f"{SETTINGS.clearml.project} raw data",
            dataset_tags=SETTINGS.clearml.tags,
        )
        local_dataset.add_files(path=SETTINGS.storage.raw_folder)

        dataset_id = None
        if remote_dataset:
            removed_files = local_dataset.list_removed_files(dataset_id=remote_dataset.id)
            modified_files = local_dataset.list_modified_files(dataset_id=remote_dataset.id)
            added_files = local_dataset.list_added_files(dataset_id=remote_dataset.id)

            if any([removed_files, modified_files, added_files]):
                local_dataset.finalize(auto_upload=True)
                dataset_id = local_dataset.id
            else:
                Dataset.delete(dataset_id=local_dataset.id, delete_files=True)
                dataset_id = remote_dataset.id
        else:
            local_dataset.finalize(auto_upload=True)
            dataset_id = local_dataset.id

        return dataset_id
    else:
        raise PipelineExecutionError(f"Raw data folder {SETTINGS.storage.raw_folder} is empty")


def run_preprocess_step(input_dataset_id: Optional[str] = None) -> pd.DataFrame:
    if input_dataset_id:
        remote_dataset = Dataset.get(
            dataset_id=input_dataset_id,
            only_completed=True,
        )
    else:
        try:
            remote_dataset = Dataset.get(
                dataset_project=SETTINGS.clearml.project,
                dataset_name=f"{SETTINGS.clearml.project} raw data",
                dataset_tags=SETTINGS.clearml.tags,
                only_completed=True,
            )
        except ValueError:
            raise PipelineExecutionError

    _ = remote_dataset.get_mutable_local_copy(
        SETTINGS.storage.raw_folder,
        overwrite=True,
    )

    data_path = Path(os.path.join(SETTINGS.storage.raw_folder, "data.csv"))
    target_path = Path(os.path.join(SETTINGS.storage.raw_folder, "target_train.csv"))
    data = CsvLoader(path=data_path).load()
    target = CsvLoader(path=target_path).load()

    preprocessor = PreprocessPipelineStep()
    return preprocessor.start(data=data, target=target)


def run_split_dataset_step(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data.empty:
        raise PipelineExecutionError("Data is empty")
    else:
        fe = SplitDatasetPipelineStep()
        return fe.start(data=data)


def run_select_features_step(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        raise PipelineExecutionError("Data is empty")
    else:
        fe = SelectFeaturesPipelineStep()
        return fe.start(data)


def run_feature_engineer_step(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: Optional[List[str]] = None,
) -> Tuple['FeatureEngineer', pd.DataFrame, pd.DataFrame]:
    if selected_features:
        train = train[selected_features + [TARGET.name]]
        test = test[selected_features + [TARGET.name]]

    if train.empty:
        raise PipelineExecutionError("Data is empty")
    else:
        fe = FeatureEngineerPipelineStep()
        fitted_fe, train_features, test_features = fe.start(
            train=train,
            test=test,
        )
        return fitted_fe, train_features, test_features


if __name__ == '__main__':

    pipe = PipelineController(
        name=f'{SETTINGS.clearml.project} tasks pipeline',
        project=SETTINGS.clearml.project,
        version='0.0.1',
        add_pipeline_tags=False,
    )

    pipe.add_function_step(
        name=PRERUN.name,
        task_type=PRERUN.task_type,
        function=run_prerun_step,
        function_return=['dataset_id'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        )
    )

    pipe.add_function_step(
        name=PREPROCESS.name,
        task_type=PREPROCESS.task_type,
        parents=[PRERUN.name],
        function=run_preprocess_step,
        function_kwargs=dict(
            input_dataset_id='${prerun.dataset_id}'
        ),
        function_return=['preprocessed_data'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        )
    )

    pipe.add_function_step(
        name=SPLIT_DATASET.name,
        task_type=SPLIT_DATASET.task_type,
        parents=[PREPROCESS.name],
        function=run_split_dataset_step,
        function_kwargs=dict(
            data='${preprocess.preprocessed_data}'
        ),
        function_return=['train', 'test', 'unlabeled'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        ),
    )

    pipe.add_function_step(
        name=SELECT_FEATURES.name,
        task_type=SELECT_FEATURES.task_type,
        parents=[SPLIT_DATASET.name],
        function=run_select_features_step,
        function_kwargs=dict(
            data='${split_dataset.train}'
        ),
        function_return=['selected_features'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        )
    )

    pipe.add_function_step(
        name=FEATURE_ENGINEER.name,
        task_type=FEATURE_ENGINEER.task_type,
        parents=[SELECT_FEATURES.name],
        function=run_feature_engineer_step,
        function_kwargs=dict(
            train='${split_dataset.train}',
            test='${split_dataset.test}',   
            selected_features='${select_features.selected_features}'
        ),
        function_return=['feature_engineer', 'train', 'test'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        )
    )

    pipe.set_default_execution_queue(SETTINGS.clearml.queue_name)
    if SETTINGS.clearml.execute_remotely:
        # Starting the pipeline (in the background)
        pipe.start()
    else:
        # for debugging purposes use local jobs
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished")
