# -*- coding: utf-8 -*-
""" Base transformer """
from abc import abstractmethod

from clearml import Task
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    r"""Abstract class for all transformers."""
    def fit(self, X, y=None):
        return self
    
    @abstractmethod
    def transform(self):
        pass