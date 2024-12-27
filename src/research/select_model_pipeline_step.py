# -*- coding: utf-8 -*-
import logging
from typing import Any, Tuple, Union
import warnings

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from pandas.util import hash_array
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from common.features import TARGET
from common.pipeline_steps import SELECT_MODEL
from core import BasePipelineStep

warnings.simplefilter(action='ignore', category=FutureWarning)


class SelectModelPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(SELECT_MODEL)

        self.scoring: str = self.step_params.get('scoring', 'balanced_accuracy')
        self.n_splits: int = self.step_params.get('n_splits', 5)
        self.n_estimators: int = self.step_params.get('n_estimators', 300)
        self.subsample: float = self.step_params.get('subsample', 0.8)
        self.test_size: float = self.step_params.get('test_size', 0.2)

    def _columns_mapping(self, data: pd.DataFrame) -> dict:
        columns_mapping = dict(zip(
            data.columns.values,
            hash_array(data.columns.values, encoding='utf8')
        ))
        return columns_mapping

    @property
    def _param_grid(self) -> dict[str, Any]:
        return [
            {
                'classifier': [
                    GradientBoostingClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.settings.random_seed,
                        subsample=self.subsample,
                    )
                ],
                'classifier__max_depth': [4, 6, 8],
                'preprocessor__scaler': [None],
            },
            {
                'classifier': [
                    LGBMClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.settings.random_seed,
                        subsample=self.subsample,
                        verbose=-1,
                    )
                ],
                'classifier__max_depth': [4, 6, 8],
                'preprocessor__scaler': [None],
            },
            {
                'classifier': [
                    CatBoostClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.settings.random_seed,
                        subsample=self.subsample,
                        verbose=False,
                    )
                ],
                'classifier__max_depth': [4, 6, 8],
                'preprocessor__scaler': [None],
            },
            {
                'classifier': [
                    LogisticRegression(
                        solver='lbfgs',
                        max_iter=1000,
                    )
                ],
                'classifier__C': [0.05, 0.01],
                'preprocessor__scaler': [None],
            },
            {
                'classifier': [
                    LogisticRegression(
                        solver='lbfgs',
                        max_iter=1000,
                    )
                ],
                'classifier__C': [0.05, 0.01],
            }
        ]

    @property
    def _splitter(self) -> dict[str, Any]:
        return StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.settings.random_seed,
        )

    @property
    def _preprocessor(self) -> 'Pipeline':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    def start(
        self,
        data: pd.DataFrame,
    ) -> Tuple[Any, dict[str, Union[str, int, float]]]:

        columns_mapping = self._columns_mapping(data)
        y = data[TARGET.name]
        X = data.drop(columns=[TARGET.name]).rename(columns=columns_mapping)

        pipe = Pipeline([
            ('preprocessor', self._preprocessor),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
            ))
        ])

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=self._param_grid,
            scoring=self.scoring,
            cv=self._splitter,
            error_score='raise',
        )
        gs.fit(X, y)

        #  report metrics
        self.task.logger.report_text(
            f'Best params: {gs.best_params_}',
            level=logging.DEBUG,
            print_console=False
        )
        self.task.upload_artifact(
            'CV metrics',
            artifact_object=pd.DataFrame(gs.cv_results_)
        )
        self.task.logger.report_single_value(
            name=f'train best {self.scoring} score',
            value=gs.best_score_,
        )

        best_model = gs.best_estimator_
        params = [k for k in gs.best_params_ if 'classifier__' in k.lower()]
        best_params = {k.split('classifier__')[1]: gs.best_params_[k] for k in params}

        return best_model, best_params
