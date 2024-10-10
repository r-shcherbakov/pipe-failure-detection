from clearml.automation import PipelineController

from common.enums import PipelineSteps
from settings import ExperimentSettings


settings = ExperimentSettings()

def post_execute_callback(a_pipeline: PipelineController, a_node: PipelineController.Node) -> None:
    print('Completed Task id={}'.format(a_node.executed))
    
    
# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name=f'{settings.clearml.project} tasks pipeline', 
    project=settings.clearml.project, 
    version='0.0.1',
    add_pipeline_tags=False,
    retry_on_failure=3,
    auto_version_bump=True,
)

pipe.add_step(
    name=f'{PipelineSteps.preprocess} step',
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.preprocess} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name=f'{PipelineSteps.feature_engineer} step',
    parents=[f'{PipelineSteps.preprocess} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.feature_engineer} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name=f'{PipelineSteps.split_dataset} step',
    parents=[f'{PipelineSteps.feature_engineer} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.split_dataset} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
    retry_on_failure=5,
)

pipe.add_step(
    name=f'{PipelineSteps.train} step',
    parents=[f'{PipelineSteps.split_dataset} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.train} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.add_step(
    name=f'{PipelineSteps.plotting} step',
    parents=[f'{PipelineSteps.train} step'],
    base_task_project=settings.clearml.project,
    base_task_name=f'{PipelineSteps.plotting} task',
    cache_executed_step=True,
    post_execute_callback=post_execute_callback,
)

pipe.set_default_execution_queue("default")
if settings.clearml.execute_remotely:
    # Starting the pipeline (in the background)
    pipe.start()
else:
    # for debugging purposes use local jobs
    pipe.start_locally(run_pipeline_steps_locally=True)

print("Pipeline successfully finished")