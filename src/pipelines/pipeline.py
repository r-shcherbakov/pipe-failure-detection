from clearml.automation import PipelineController

from common.pipeline_steps import (
    PREPROCESS,
    FEATURE_ENGINEER,
    SPLIT_DATASET,
    TRAIN,
    PLOTTING,
)
from settings import ClearmlSettings

settings = ClearmlSettings()


def post_execute_callback(
    a_pipeline: PipelineController,
    a_node: PipelineController.Node
) -> None:
    print('Completed Task id={}'.format(a_node.executed))


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name=f'{settings.project} tasks pipeline',
    project=settings.project,
    version='0.0.1',
    add_pipeline_tags=False,
    retry_on_failure=3,
    auto_version_bump=True,
)

pipe.add_step(
    name=PREPROCESS.name,
    base_task_project=settings.project,
    base_task_name=f'{PREPROCESS.name} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=2,
)

pipe.add_step(
    name=FEATURE_ENGINEER.name,
    parents=[PREPROCESS.name],
    base_task_project=settings.project,
    base_task_name=f'{FEATURE_ENGINEER.name} task',
    parameter_override={
        "General/input_dataset_id": "${preprocess.parameters.General/output_dataset_id}",
    },
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=1,
)

pipe.add_step(
    name=SPLIT_DATASET.name,
    parents=[FEATURE_ENGINEER.name],
    base_task_project=settings.project,
    base_task_name=f'{SPLIT_DATASET.name} task',
    parameter_override={
        "General/input_dataset_id": "${feature_engineer.parameters.General/output_dataset_id}",
    },
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.add_step(
    name=TRAIN.name,
    parents=[FEATURE_ENGINEER.name],
    base_task_project=settings.project,
    base_task_name=f'{TRAIN.name} task',
    parameter_override={
        "General/input_dataset_id": "${split_dataset.parameters.General/output_dataset_id}",
    },
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.add_step(
    name=PLOTTING.name,
    parents=[TRAIN.name],
    base_task_project=settings.project,
    base_task_name=f'{PLOTTING.name} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.set_default_execution_queue(settings.queue_name)
if settings.execute_remotely:
    # Starting the pipeline (in the background)
    pipe.start()
else:
    # for debugging purposes use local jobs
    pipe.start_locally(run_pipeline_steps_locally=True)

print("Pipeline successfully finished")
