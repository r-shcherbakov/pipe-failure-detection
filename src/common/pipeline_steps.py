from dataclasses import dataclass

from clearml import TaskTypes


@dataclass(frozen=True)
class PipelineStep:
    """Dataclass for describing pipeline steps"""

    name: str
    task_type: str

    def __str__(self):
        return self.name


PRERUN = PipelineStep(
    name="prerun",
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
    task_type=TaskTypes.data_processing.name,
)
SELECT_MODEL = PipelineStep(
    name="select_model",
    task_type=TaskTypes.optimizer.name,
)
MODELWISE_ANALYSIS = PipelineStep(
    name="modelwise_analysis",
    task_type=TaskTypes.optimizer.name,
)
SAMPLEWISE_ANALYSIS = PipelineStep(
    name="samplewise_analysis",
    task_type=TaskTypes.optimizer.name,
)
HYPERPARAMETER_OPTIMIZATION = PipelineStep(
    name="hyperparameter_optimization",
    task_type=TaskTypes.optimizer.name
)
TRAIN = PipelineStep(
    name="train",
    task_type=TaskTypes.training.name,
)
PLOTTING = PipelineStep(
    name="plotting",
    task_type=TaskTypes.service.name,
)
POSTRUN = PipelineStep(
    name="postrun",
    task_type=TaskTypes.service.name
)
