# -*- coding: utf-8 -*-
""" Base pipeline step """
from abc import ABC, abstractmethod
import logging
import traceback
from typing import Any, Dict, Optional, TYPE_CHECKING

from clearml import Task
import yaml

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep
    from settings import Settings


class BasePipelineStep(ABC):
    r"""Abstract class for all pipeline steps."""

    def __init__(
        self,
        settings: 'Settings',
        pipeline_step: 'PipelineStep',
    ):
        self.settings: 'Settings' = settings
        self.pipeline_step: 'PipelineStep' = pipeline_step

        self._init_task()
        self._init_parameters()

    def _init_task(self):
        self.task: Task = Task.init(
            project_name=self.settings.clearml.project,
            task_name=f'{self.pipeline_step.name} task',
            task_type=self.pipeline_step.task_type,
            tags=self.settings.clearml.tags,
            deferred_init=True,
            reuse_last_task_id=False)
        if self.settings.clearml.execute_remotely:
            self.task.execute_remotely(queue_name=self.settings.clearml.queue_name)

    def _init_parameters(self):
        with open(self.settings.params_path) as file:
            params = yaml.load(file, Loader=yaml.Loader)
            self.common_params: Optional[Dict[str, Any]] = params.get('common', None)
            self.step_params = params.get(self.pipeline_step.name, None)
        if self.common_params:
            self.task.connect(self.common_params, name="common")
        if self.step_params:
            self.task.connect(self.step_params, name=self.pipeline_step.name.replace('_', ' '))

    def _log_success_step_execution(self) -> None:
        self.task.logger.report_text(
            f"Execution {self.pipeline_step.name.replace('_', ' ')} successfully finished",
            level=logging.INFO
        )

    def _log_failed_step_execution(self, exception: Exception) -> None:
        self.task.logger.report_text(
            f"Execution {self.pipeline_step.name.replace('_', ' ')} for  failed due to: {exception}",
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(),
            level=logging.DEBUG,
            print_console=False,
        )

    @abstractmethod
    def start(self):
        pass
