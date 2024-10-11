from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

from clearml import TaskTypes
from settings import StorageSettings, ArtifactsSettings

storage_settings = StorageSettings()
artifacts_settings = ArtifactsSettings()

@dataclass(frozen=True)
class PipelineStep:
    """Dataclass for describing pipeline steps"""

    name: str
    task_type: str
    input_directory: Optional[Union[str, Path]] = None
    output_directory: Optional[Union[str, Path]] = None

    def __str__(self):
        return self.name
    
    
PRE_RUN = PipelineStep(
    name="pre_run",
    task_type=TaskTypes.service.name,
    input_directory=storage_settings.raw_folder,
)
PREPROCESS = PipelineStep(
    name="preprocess",
    task_type=TaskTypes.data_processing.name,
    input_directory=storage_settings.raw_folder,
    output_directory=storage_settings.processed_folder
)
SPLIT_DATASET = PipelineStep(
    name="split_dataset",
    task_type=TaskTypes.data_processing.name,
    input_directory=storage_settings.processed_folder,
    output_directory=storage_settings.splitted_folder
)
FEATURE_ENGINEER = PipelineStep(
    name="feature_engineer",
    task_type=TaskTypes.data_processing.name,
    input_directory=storage_settings.splitted_folder,
    output_directory=storage_settings.features_folder
)
SELECT_FEATURES = PipelineStep(
    name="select_features",
    task_type=TaskTypes.data_processing.name
)
TRAIN = PipelineStep(
    name="train",
    task_type=TaskTypes.training.name,
    input_directory=storage_settings.features_folder,
    output_directory=storage_settings.prediction_folder
)
PLOTTING = PipelineStep(
    name="plotting",
    task_type=TaskTypes.service.name,
    input_directory=storage_settings.prediction_folder,
    output_directory=artifacts_settings.plots_folder
)
HYPERPARAMETER_OPTIMIZATION = PipelineStep(
    name="hyperparameter_optimization",
    task_type=TaskTypes.optimizer.name
)
POST_RUN = PipelineStep(
    name="post_run",
    task_type=TaskTypes.service.name 
)