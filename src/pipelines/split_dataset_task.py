# -*- coding: utf-8 -*-
from features import SplitDatasetPipelineStep
from settings import Settings

settings = Settings()


if __name__ == "__main__":
    SplitDatasetPipelineStep(settings=Settings())
    