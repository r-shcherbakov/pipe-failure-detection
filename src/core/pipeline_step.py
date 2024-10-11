# -*- coding: utf-8 -*-
""" Base pipeline step """
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import traceback
from typing import Any, Dict, Optional, Union

from clearml import Task, Dataset
import pandas as pd
import yaml

from common.pipeline_steps import PipelineStep
from common.exceptions import (
    DatasetDownloadError,
    PipelineExecutionError,
)
from settings import Settings
from utilities.utils import compress_pickle
from utilities.path_utils import is_empty_dir


class BasePipelineStep(ABC):
    r"""Abstract class for all pipeline steps."""

    def __init__(
        self,
        settings: 'Settings',
        pipeline_step: 'PipelineStep',
        previous_pipeline_step: 'PipelineStep',
    ):
        self.settings: Settings = settings
        self.pipeline_step: PipelineStep = pipeline_step
        self.previous_pipeline_step: PipelineStep = previous_pipeline_step

        self._init_task()
        self._init_parameters()
        self._execute()
    
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
        general_parameter = {
            'input_dataset_id': None,
        }
        self.task.connect(general_parameter)
        
        with open(self.settings.params_path) as file:
            params = yaml.load(file, Loader=yaml.Loader)
            self.common_params: Optional[Dict[str, Any]] = params.get('common', None)
            self.step_params = params.get(self.pipeline_step.name, None)
        if self.common_params:
            self.task.connect(self.common_params, name="common")  
        if self.step_params:
            self.task.connect(self.step_params, name=self.pipeline_step.name.replace('_', ' '))
        
    def _log_success_step_execution(
        self,
        file_name: str,
    ) -> None:
        self.task.logger.report_text(
            f"Execution {self.pipeline_step.name.replace('_', ' ')} for {file_name} successfully finished", 
            level=logging.INFO
        )
        
    def _log_failed_step_execution(
        self,
        file_name: str,
        exception: Exception,
    ) -> None:
        self.task.logger.report_text(
            f"Execution {self.pipeline_step.name.replace('_', ' ')} for {file_name} failed due to: {exception}", 
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(), 
            level=logging.DEBUG,
            print_console=False,
        )
        
    def _log_success_upload_dataset(self) -> None:
        self.task.logger.report_text(
            f"Dataset of {self.pipeline_step.name.replace('_', ' ')} step "
            f"successfully uploaded", 
            level=logging.INFO
        )
        
    def _log_failed_upload_dataset(
        self,
        exception: Exception,
    ) -> None:
        self.task.logger.report_text(
            f"Uploading dataset of {self.pipeline_step.name.replace('_', ' ')} step "
            f"failed due to: {exception}", 
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(), 
            level=logging.DEBUG,
            print_console=False,
        )
        
    def _log_success_save_data(
        self,
        file_name: str,
    ) -> None:
        self.task.logger.report_text(
            f"Data of {self.pipeline_step.name.replace('_', ' ')} step "
            f"for {file_name} successfully locally saved", 
            level=logging.INFO
        )
        
    def _log_failed_save_data(
        self,
        file_name: str,
        exception: Exception,
    ) -> None:
        self.task.logger.report_text(
            f"Savings data of {self.pipeline_step.name.replace('_', ' ')} step "
            f"for {file_name} failed due to: {exception}", 
            level=logging.INFO
        )
        self.task.logger.report_text(
            'traceback:' + traceback.format_exc(), 
            level=logging.DEBUG,
            print_console=False,
        )
    
    def _download_input_dataset(self) -> None:
        if dataset_id := self.task.get_parameter('General/input_dataset_id'):
            remote_dataset = Dataset.get(
                dataset_id=dataset_id,
                only_completed=True,
            )
        elif self.previous_pipeline_step:
            try:
                remote_dataset = Dataset.get(
                    dataset_project=self.settings.clearml.project,
                    dataset_name=f"{self.settings.clearml.project} "
                        f"{self.previous_pipeline_step.name.replace('_', ' ')} output dataset",
                    dataset_tags=self.settings.clearml.tags,
                    only_completed=True,
                )
            except ValueError:
                raise DatasetDownloadError
        else:
            raise PipelineExecutionError
        
        _ = remote_dataset.get_mutable_local_copy(self._input_directory)
        
    def _upload_output_dataset(self) -> None:
        dataset = Dataset.create(
            dataset_project=self.settings.clearml.project,
            dataset_name=f"{self.settings.clearml.project} {self.pipeline_step.name.replace('_', ' ')} "
                f"output dataset",
            dataset_tags=self.settings.clearml.tags,
        )  
        dataset.add_files(path=self._output_directory)
        dataset.finalize(auto_upload=True)
        self.task.set_parameter(
            name="output_dataset_id",
            value=dataset.id,
        )
        
    def _save_locally_data(
        self,
        path: Union[str, Path],
        data: pd.DataFrame,
    ) -> None:
        try:    
            file_name = compress_pickle(path, data).name
            self._log_success_save_data(file_name=file_name)
        except Exception as exception:
            self._log_failed_save_data(
                file_name=file_name,
                exception=exception,
            )
            raise PipelineExecutionError
                
    @property 
    @abstractmethod
    def _input_directory(self):
        pass
    
    def _check_input_directory(self):
        if is_empty_dir(self._input_directory):
            self._download_input_dataset()

    @property 
    @abstractmethod
    def _input_files(self):
        pass
    
    @property 
    @abstractmethod
    def _output_directory(self):
        pass
    
    @abstractmethod
    def _upload_artifacts(self) -> None:
        pass
    
    @abstractmethod
    def _process_data(self):
        pass
    
    def _execute(self):
        self._process_data()
        self._upload_output_dataset()
        self._upload_artifacts()
        
        self.task.logger.report_text(
            f"{self.pipeline_step.name} is finished",
            level=logging.INFO
        ) 