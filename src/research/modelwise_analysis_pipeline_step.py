# -*- coding: utf-8 -*-
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.util import hash_array
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
)

from src.common.features import TARGET
from src.common.pipeline_steps import MODELWISE_ANALYSIS
from src.core import BasePipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)


class ModelwiseAnalysisPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(MODELWISE_ANALYSIS)

        self.scoring: str = self.step_params.get('scoring', 'balanced_accuracy')
        self.n_splits: int = self.step_params.get('n_splits', 5)
        self.test_size: float = self.step_params.get('test_size', 0.2)

    def _columns_mapping(self, data: pd.DataFrame) -> dict:
        columns_mapping = dict(zip(
            data.columns.values,
            hash_array(data.columns.values, encoding='utf8')
        ))
        return columns_mapping

    @property
    def _param_grid(self) -> dict[str, Any]:
        return {
            'classifier__max_depth': [4, 6, 8],
            'classifier__n_estimators': [10, 30, 50, 100, 200],
            'classifier__subsample': [0.7, 0.9, 1.0],
        }

    @property
    def _splitter(self) -> dict[str, Any]:
        return StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.settings.random_seed,
        )

    def pooled_var(self, stds: pd.Series) -> float:
        n = self.n_splits # size of each group
        return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))

    def _report_analysis_plot(self, data: pd.DataFrame) -> None:
        parameters = [
            'mean_test_score',
            'mean_train_score',
            'std_test_score',
            'std_train_score',
        ]

        fig, axes = plt.subplots(1, len(self._param_grid),
                                figsize = (8*len(self._param_grid), 10),
                                sharey='row')
        axes[0].set_ylabel("Score", fontsize=25)
        lw = 2

        for idx, (param_name, param_range) in enumerate(self._param_grid.items()):
            grouped_df = data.groupby(f'param_{param_name}')[parameters] \
                .agg({
                    'mean_train_score': 'mean',
                    'mean_test_score': 'mean',
                    'std_train_score': self.pooled_var,
                    'std_test_score': self.pooled_var,
                })

            axes[idx].set_xlabel(param_name, fontsize=25)
            axes[idx].set_ylim(0.0, 1.1)
            axes[idx].plot(
                param_range,
                grouped_df['mean_train_score'],
                label="Training score",
                color="darkorange",
                lw=lw
            )
            axes[idx].fill_between(
                param_range,
                grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                grouped_df['mean_train_score'] + grouped_df['std_train_score'],
                alpha=0.2,
                color="darkorange",
                lw=lw
            )
            axes[idx].plot(
                param_range,
                grouped_df['mean_test_score'],
                label="Cross-validation score",
                color="navy",
                lw=lw
            )
            axes[idx].fill_between(
                param_range,
                grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                grouped_df['mean_test_score'] + grouped_df['std_test_score'],
                alpha=0.2,
                color="navy",
                lw=lw
            )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)
        fig.subplots_adjust(bottom=0.25, top=0.85)

        self.task.logger.report_matplotlib_figure(
            title="Validation curves",
            series="Interactive plot",
            iteration=0,
            figure=plt,
            report_interactive=True,
        )

    def start(self, model: Any, data: pd.DataFrame) -> None:

        columns_mapping = self._columns_mapping(data)
        y = data[TARGET.name]
        X = data.drop(columns=[TARGET.name]).rename(columns=columns_mapping)

        gs = GridSearchCV(
            estimator=model,
            param_grid=self._param_grid,
            scoring=self.scoring,
            cv=self._splitter,
            error_score='raise',
            return_train_score=True,
        )
        gs.fit(X, y)

        # report config
        self.task.upload_artifact('best_params', gs.best_params_)

        # report plots
        self._report_analysis_plot(pd.DataFrame(gs.cv_results_))
