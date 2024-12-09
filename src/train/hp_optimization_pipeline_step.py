# -*- coding: utf-8 -*-
import logging
from typing import Any, Union, TYPE_CHECKING
import warnings

from catboost import CatBoostClassifier
import optuna
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
)

from common.features import TARGET
from common.pipeline_steps import HYPERPARAMETER_OPTIMIZATION
from core import BasePipelineStep

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)


class HPOptimizationPipelineStep(BasePipelineStep):
    def __init__(self):
        self.pipeline_step: 'PipelineStep' = HYPERPARAMETER_OPTIMIZATION
        super().__init__(self.pipeline_step)

        self.n_trials: int = self.n_trials.get('n_splits', 50)
        self.n_splits: int = self.step_params.get('n_splits', 5)
        self.test_size: float = self.step_params.get('test_size', 0.2)
        self.scoring: str = self.step_params.get('scoring', 'balanced_accuracy')

    @property
    def _splitter(self) -> dict[str, Any]:
        return StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.settings.random_seed,
        )

    def _objective(self, trial) -> float:
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        estimator = CatBoostClassifier(**param)
        cv_score = cross_val_score(
            estimator=estimator,
            X=self.data.drop(columns=[TARGET.name]),
            y=self.data[TARGET.name],
            cv=self._splitter,
            scoring=self.scoring,
        )

        return cv_score.mean()

    def start(self, data: pd.DataFrame) ->  dict[str, Union[str, int, float]]:
        self.data = data

        study = optuna.create_study(direction="maximize")
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
        )

        self.task.logger.report_text(
            f"Number of finished trials: {len(study.trials)}",
            level=logging.INFO
        )
        trial = study.best_trial
        self.task.logger.report_single_value(
            name=f'HP optimization best {self.scoring} score',
            value=trial.value,
        )

        # TODO: report params
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
