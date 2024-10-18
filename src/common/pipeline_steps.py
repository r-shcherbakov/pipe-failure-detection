from dataclasses import dataclass

from clearml import TaskTypes


@dataclass(frozen=True)
class PipelineStep:
    """Dataclass for describing pipeline steps"""

    name: str
    task_type: str

    def __str__(self):
        return self.name


PRE_RUN = PipelineStep(
    name="pre_run",
    task_type=TaskTypes.service.name,
)
PREPROCESS = PipelineStep(
    name="preprocess",
    task_type=TaskTypes.data_processing.name,
)
SPLIT_DATASET = PipelineStep(
    name="split_dataset",
    task_type=TaskTypes.data_processing.name,
)
FEATURE_ENGINEER = PipelineStep(
    name="feature_engineer",
    task_type=TaskTypes.data_processing.name,
)
SELECT_FEATURES = PipelineStep(
    name="select_features",
    task_type=TaskTypes.data_processing.name
)
TRAIN = PipelineStep(
    name="train",
    task_type=TaskTypes.training.name,
)
PLOTTING = PipelineStep(
    name="plotting",
    task_type=TaskTypes.service.name,
)
HYPERPARAMETER_OPTIMIZATION = PipelineStep(
    name="hyperparameter_optimization",
    task_type=TaskTypes.optimizer.name
)
POST_RUN = PipelineStep(
    name="post_run",
    task_type=TaskTypes.service.name
)
