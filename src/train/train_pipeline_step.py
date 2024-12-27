# -*- coding: utf-8 -*-
import os
import gc
from joblib import dump
import logging
from typing import Optional, Union, TYPE_CHECKING
import warnings

from catboost import (
    Pool,
    CatBoostClassifier,
)
from clearml import OutputModel
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from core import BasePipelineStep
from common.enums import DefectType
from common.pipeline_steps import TRAIN
from common.features import TARGET
from preprocess.preprocessor import Preprocessor

if TYPE_CHECKING:
    from features.feature_engineer import FeatureEngineer

warnings.simplefilter(action="ignore", category=FutureWarning)


class TrainPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(TRAIN)

    @property
    def _metrics(self) -> list[str]:
        return ["Precision", "Recall", "F1", "BalancedAccuracy",  "AUC"]

    def _get_metrics(
        self,
        model: CatBoostClassifier,
        eval_pool: Pool
    ) -> dict[str, float]:
        metrics = {}
        for metric in self._metrics:
            metrics[f"{metric}"] = model.eval_metrics(
                data=eval_pool,
                metrics=[metric]
            )[metric][-1]
        return metrics

    def _get_cb_pool(self, data: Optional[pd.DataFrame]) -> Pool:
        if not data.empty:
            X = data.drop(axis=1, columns=TARGET.name).copy()
            y = data[TARGET.name].fillna(0).copy()
            cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

            pool = Pool(
                data=X,
                label=y,
                cat_features=cat_features,
            )
            return pool
        else:
            return None

    def _log_cv_metrics(self) -> None:
        """
        Uploaded results of common metrics for each fold
        provided by GroupKFold cross-validation.
        """

        X = self.train_data.drop(axis=1, columns=TARGET.name).copy()
        y = self.train_data[TARGET.name].fillna(0).copy()

        n_splits = self.params.pop("n_splits", 5)
        test_size: float = self.params.pop('test_size', 0.2)
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=self.settings.random_seed,
        )
        cv_result = pd.DataFrame()
        for fold_num, (train_idx, valid_idx) in enumerate(tqdm(splitter.split(X, y), total=n_splits)):
            train_pool = self._get_cb_pool(self.train_data.iloc[train_idx])
            eval_pool = self._get_cb_pool(self.train_data.iloc[valid_idx])

            model = CatBoostClassifier(
                **self.params,
                random_seed=self.settings.random_seed,
                silent=True,
            ).fit(
                train_pool,
                eval_set=eval_pool,
            )

            fold_result = pd.DataFrame(
                data=self._get_metrics(model, eval_pool),
                index=[fold_num]
            )
            cv_result = pd.concat([cv_result, fold_result])
            del train_pool, eval_pool, model
            gc.collect()

        cv_result.loc["mean"] = cv_result.mean()
        cv_result = cv_result.reset_index(names=['fold_num'])
        self.task.logger.report_table(
            title="CV results",
            series="CV results",
            table_plot=cv_result
        )

    def _log_test_metrics(self) -> None:
        train_pool = self._get_cb_pool(self.train_data)
        test_pool = self._get_cb_pool(self.test_data)

        model = CatBoostClassifier(
            **self.params,
            random_seed=self.settings.random_seed,
            silent=True,
        ).fit(
            train_pool,
            eval_set=test_pool,
        )

        # Log test metrics
        if test_pool:
            test_metrics = pd.DataFrame(
                data=self._get_metrics(model, test_pool),
                index=[0]
            )
            self.task.logger.report_table(
                title="test metrics",
                series="test metrics",
                table_plot=test_metrics
            )

        del train_pool, test_pool
        gc.collect()

    def _train_model(
        self,
        feature_engineer: 'FeatureEngineer',
        selected_features: list[str]
    ) -> Pipeline:
        overall_data = pd.concat([self.train_data, self.test_data])
        train_pool = self._get_cb_pool(overall_data)
        model = CatBoostClassifier(
            **self.params,
            random_seed=self.settings.random_seed,
        ).fit(train_pool, verbose=True)

        pipeline = Pipeline([
            ('preprocessor', Preprocessor(
                selected_features=selected_features
            )),
            ('feature_engineer', feature_engineer),
            ('model', model)
        ])
        model_filepath = os.path.join(
            self.settings.artifacts.models_folder,
            "release_model.pkl"
        )
        dump(
            value=pipeline,
            filename=model_filepath,
            compress=9
        )

    def start(
        self,
        train: pd.DataFrame,
        test: Optional[pd.DataFrame],
        params: dict[str, Union[str, int, float]],
        feature_engineer: 'FeatureEngineer',
        selected_features: list[str],
    ) -> None:
        self.train_data = train
        self.test_data = test
        # Be carefull with it because best params will be overwritten by manual step params
        self.params = params | self.step_params

        # Cross validation
        self._log_cv_metrics()

        # Train final model
        self._log_test_metrics()

        # TODO: Create sklearn pipeline with Preprocessor, FeatureEngineer and Model
        # as overall pipeline
        self._train_model(
          feature_engineer=feature_engineer,
          selected_features=selected_features
        )
