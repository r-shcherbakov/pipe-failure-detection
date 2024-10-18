from clearml import PipelineController

from common.pipeline_steps import (
    PREPROCESS,
    FEATURE_ENGINEER,
)
from settings import Settings
from preprocess import run_preprocess_step
from features.feature_engineer_pipeline_step import run_feature_engineer_step


settings = Settings()


if __name__ == '__main__':

    pipe = PipelineController(
        name=f'{settings.clearml.project} tasks pipeline',
        project=settings.clearml.project,
        version='0.0.1',
        add_pipeline_tags=False,
        retry_on_failure=3,
        auto_version_bump=True,
    )

    pipe.add_function_step(
        name=PREPROCESS.name,
        task_type=PREPROCESS.task_type,
        function=run_preprocess_step,
        function_kwargs=dict(settings=settings),
        function_return=['preprocessed_data'],
        cache_executed_step=True,
        retry_on_failure=2,

    )
    pipe.add_function_step(
        name=FEATURE_ENGINEER.name,
        task_type=FEATURE_ENGINEER.task_type,
        function=run_feature_engineer_step,
        function_kwargs=dict(settings=settings, data='${preprocess.preprocessed_data}'),
        function_return=['features'],
        cache_executed_step=True,
    )
    # pipe.add_function_step(
    #     name='step_three',
    #     # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
    #     function=step_three,
    #     function_kwargs=dict(data='${step_two.processed_data}'),
    #     function_return=['model'],
    #     cache_executed_step=True,
    # )

    pipe.set_default_execution_queue(settings.clearml.queue_name)
    if settings.clearml.execute_remotely:
        # Starting the pipeline (in the background)
        pipe.start()
    else:
        # for debugging purposes use local jobs
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline successfully finished")
