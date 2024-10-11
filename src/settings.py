# -*- coding: utf-8 -*-
'''Main settings'''
import os
import logging
from pathlib import Path
import random
from typing import Annotated, List, Union

import numpy as np
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    Field,
    computed_field,
    field_validator
)
from pydantic_settings import BaseSettings

from utilities.logging import set_logging

PROJECT_PATH = Path(__file__).resolve().parents[1]


def fix_seed(random_seed: int = 42) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)


def create_missed_directory(directory: Union[str, Path]) -> Path:
    if not isinstance(directory, Path):
        directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    return directory


class ArtifactsSettings(BaseModel):
    root_folder: DirectoryPath = Field(
        Path(os.path.join(PROJECT_PATH, "artifacts")), 
        description="Path to the local artifacts folder", 
        validate_default=True)
    
    @field_validator("root_folder", mode="before")
    def validate_root_folder(cls, directory: Union[str, Path]) -> str:
        directory = create_missed_directory(directory)
        return directory
    
    @computed_field(description="Path to the fitted models")
    def models_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "models"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
    
    @computed_field(description="Path to the reports")
    def reports_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "reports"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
        
    @computed_field(description="Path to the plots")
    def plots_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "plots"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
        

class StorageSettings(BaseModel):
    root_folder: DirectoryPath = Field(
        Path(os.path.join(PROJECT_PATH, "data")), 
        description="Path to the mounted dataset storage", 
        validate_default=True)
    
    @field_validator("root_folder", mode="before")
    def validate_root_folder(cls, directory: Union[str, Path]) -> str:
        directory = create_missed_directory(directory)
        return directory
    
    @computed_field(description="Path to the raw pipeline data")
    def raw_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "raw"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
    
    @computed_field(description="Path to the external data for pipeline (labels and etc.)")
    def external_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "external"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
    
    @computed_field(description="Path to the processed pipeline data")
    def processed_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "processed"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory
    
    @computed_field(description="Path to the pipeline data with features")
    def features_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "features"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory  

    @computed_field(description="Path to the splitted pipeline data")
    def splitted_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "splitted"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory  

    @computed_field(description="Path to the train pipeline data")
    def train_folder(self) -> Path:
        directory = Path(os.path.join(self.splitted_folder, "train"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory    
    
    @computed_field(description="Path to the test pipeline data")
    def test_folder(self) -> Path:
        directory = Path(os.path.join(self.splitted_folder, "test"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory   
    
    @computed_field(description="Path to the model's predictions")
    def prediction_folder(self) -> Path:
        directory = Path(os.path.join(self.root_folder, "prediction"))
        directory.mkdir(exist_ok=True, parents=True)
        return directory  
    
    
class ClearmlSettings(BaseModel):
    execute_remotely: bool = Field(False, description='Option to enqueue task for remote execution')
    queue_name: str = Field('default', description='The name of the queue')
    project: str = Field("Pipe failure detection", description='Project name for the tasks')
    tags: List[str] = Field(
        ["Pipe failure detection"], 
        description=' A list of tags which describe the Task to add'
    )
    # output_url: str = Field('s3://bucket/data', description='Target storage for the compressed dataset')


class ParallelbarSettings(BaseModel):
    n_cpu: int = Field(3)
    process_timeout: int = Field(3600, description='Timeout of one process in seconds')
    error_behavior: str = Field('coerce', description='Specifies what to do upon encountering an error')
    
    
class LoggingSettings(BaseModel):
    level: int = Field(logging.INFO, description='Timeout of one process in seconds')


class Settings(BaseSettings):
    params_path: FilePath = Field(
        os.path.join(Path(__file__).resolve().parent, 'params.yaml'), 
        description='Path to the experiment parameters config'
    )
    random_seed: int = Field(42, description='Seed for equivalent experiment results')
    
    clearml: ClearmlSettings = Field(default_factory=ClearmlSettings)
    parallelbar: ParallelbarSettings = Field(default_factory=ParallelbarSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    artifacts: ArtifactsSettings = Field(default_factory=ArtifactsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        env_file = os.getenv('ENV', '.env')
        env_nested_delimiter = '__'
        
    @field_validator('random_seed', mode='after')
    def fix_seed(cls, random_seed):
        fix_seed(random_seed)
        return random_seed
        
    @field_validator('logging', mode='after')
    def set_logging(cls, logging):
        set_logging(logging.level)
        return logging
        
    
