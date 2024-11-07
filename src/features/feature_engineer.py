# -*- coding: utf-8 -*-
""" Base feature engineer """
import logging

import pandas as pd
from sklearn import set_config
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import BaseTransformer

LOGGER = logging.getLogger(__name__)


class FeatureEngineer(BaseTransformer):
    def __init__(
        self,
        pca_n_components: int,
        random_state: int,
        kmeans_n_clusters: int,
        kmeans_init: str,
    ):
        super().__init__()
        self.pca_n_components = pca_n_components
        self.random_state = random_state
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init

    def fit(self, X, y=None):
        pipeline = Pipeline([
            ("scaler", StandardScaler(
                with_mean=True,
            )),
            ("pca", PCA(
                n_components=self.pca_n_components,
                random_state=self.random_state,
            )),
            ("clustering", KMeans(
                init=self.kmeans_init,
                n_clusters=self.kmeans_n_clusters,
                random_state=self.random_state,
            ))
        ])
        self.prefitted_pipeline = pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        combined_pipeline = Pipeline([
            ("prefitted_pipeline", self.prefitted_pipeline),
        ])
        set_config(transform_output="pandas")
        X["cluster"] = combined_pipeline.predict(X)

        return X
