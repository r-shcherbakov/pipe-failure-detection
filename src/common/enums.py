# -*- coding: utf-8 -*-
"""Module defines enums."""


class PipelineSteps:
    """
    Enum for pipeline steps.
    """

    pre_run = 'pre_run'
    preprocess = 'preprocess'
    feature_engineer = 'feature_engineer'
    select_features = 'select_features'
    split_dataset = 'split_dataset'
    train = 'train'
    plotting = 'plotting'
    hyperparameter_optimization = 'hyperparameter_optimization'
    post_run = 'post_run'