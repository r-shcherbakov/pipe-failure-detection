import os
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

from clearml import PipelineController, Dataset
import pandas as pd

from common.exceptions import PipelineExecutionError
from common.pipeline_steps import (
    PRERUN,
    PREPROCESS,
    FEATURE_ENGINEER,
    SPLIT_DATASET,
)
from features import FeatureEngineerPipelineStep, SplitDatasetPipelineStep
from preprocess import PreprocessPipelineStep
from settings import Settings
from utilities.loaders import CsvLoader
from utilities.path_utils import is_empty_dir

if TYPE_CHECKING:
    from features.feature_engineer import FeatureEngineer


def run_prerun_step(settings: 'Settings') -> str:
    if not is_empty_dir(settings.storage.raw_folder):
        dataset = Dataset.create(
            dataset_project=settings.clearml.project,
            dataset_name=f"{settings.clearml.project} raw data",
            dataset_tags=settings.clearml.tags,
        )
        dataset.add_files(path=settings.storage.raw_folder)
        dataset.finalize(auto_upload=True)
        return dataset.id
    else:
        raise PipelineExecutionError(f"Raw data folder {settings.storage.raw_folder} is empty")


def run_preprocess_step(
    settings: 'Settings',
    input_dataset_id: Optional[str] = None,
) -> pd.DataFrame:
    if input_dataset_id:
        remote_dataset = Dataset.get(
            dataset_id=input_dataset_id,
            only_completed=True,
        )
    else:
        try:
            remote_dataset = Dataset.get(
                dataset_project=settings.clearml.project,
                dataset_name=f"{settings.clearml.project} raw data",
                dataset_tags=settings.clearml.tags,
                only_completed=True,
            )
        except ValueError:
            raise PipelineExecutionError

    _ = remote_dataset.get_mutable_local_copy(
        settings.storage.raw_folder,
        overwrite=True,
    )

    data_path = Path(os.path.join(settings.storage.raw_folder, "data.csv"))
    target_path = Path(os.path.join(settings.storage.raw_folder, "target_train.csv"))
    data = CsvLoader(path=data_path).load()
    target = CsvLoader(path=target_path).load()

    preprocessor = PreprocessPipelineStep(settings=settings)
    return preprocessor.start(data=data, target=target)


def run_split_dataset_step(
    settings: 'Settings',
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data.empty:
        raise PipelineExecutionError("Data is empty")
    else:
        fe = SplitDatasetPipelineStep(settings=settings)
        return fe.start(data=data)


def run_feature_engineer_step(
    settings: 'Settings',
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple['FeatureEngineer', pd.DataFrame, pd.DataFrame]:
    if train.empty:
        raise PipelineExecutionError("Data is empty")
    else:
        fe = FeatureEngineerPipelineStep(settings=settings)
        fitted_fe, train_features, test_features = fe.start(
            train=train,
            test=test,
        )
        return fitted_fe, train_features, test_features


if __name__ == '__main__':

    settings = Settings()
    pipe = PipelineController(
        name=f'{settings.clearml.project} tasks pipeline',
        project=settings.clearml.project,
        version='0.0.1',
        add_pipeline_tags=False,
        auto_version_bump=True,
    )

    pipe.add_function_step(
        name=PRERUN.name,
        task_type=PRERUN.task_type,
        function=run_prerun_step,
        function_kwargs=dict(settings=settings),
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
            settings=settings,
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
            settings=settings,
            data='${preprocess.preprocessed_data}'
        ),
        function_return=['train', 'test', 'unlabeled'],
        cache_executed_step=True,
        continue_behaviour=dict(
            continue_on_fail=False,
            continue_on_abort=False,
        )
    )

    # pipe.add_function_step(
    #     name=FEATURE_ENGINEER.name,
    #     task_type=FEATURE_ENGINEER.task_type,
    #     parents=[PREPROCESS.name],
    #     function=run_feature_engineer_step,
    #     function_kwargs=dict(settings=settings, data='${preprocess.preprocessed_data}'),
    #     function_return=['features'],
    #     cache_executed_step=True,
    # )

    pipe.set_default_execution_queue(settings.clearml.queue_name)
    if settings.clearml.execute_remotely:
        # Starting the pipeline (in the background)
        pipe.start()
    else:
        # for debugging purposes use local jobs
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished")
