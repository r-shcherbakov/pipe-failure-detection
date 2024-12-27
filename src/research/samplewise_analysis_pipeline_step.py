# -*- coding: utf-8 -*-
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.util import hash_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from common.features import TARGET
from common.pipeline_steps import SAMPLEWISE_ANALYSIS
from core import BasePipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)


class SamplewiseAnalysisPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(SAMPLEWISE_ANALYSIS)

        self.test_size: float = self.step_params.get('test_size', 0.2)
        self.samples: list[float] = self.step_params.get(
            'samples',
            [0.2, 0.4, 0.6, 0.8, 1]
        )

    def _columns_mapping(self, data: pd.DataFrame) -> dict:
        columns_mapping = dict(zip(
            data.columns.values,
            hash_array(data.columns.values, encoding='utf8')
        ))
        return columns_mapping

    def _sample_data(
        self,
        data: pd.DataFrame,
        sample_size: float,
        stratify_column: str,
    ) -> pd.DataFrame:
        assert 0.0 < sample_size <= 1.0
        assert stratify_column in data.columns

        frac = int(sample_size * len(data)) / len(data)
        return data.groupby(stratify_column).sample(frac=frac)

    def _report_analysis_plot(self, data: dict[float, float]) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Samplewise Analysis')
        ax.set_xlabel('Sample Size', fontsize=25)
        ax.set_ylabel('Balanced Accuracy', fontsize=25)
        ax.set_ylim(0.0, 1.05)
        ax.plot(
            data.keys(),
            data.values(),
            color="darkorange",
        )

        self.task.logger.report_matplotlib_figure(
            title="Samplewise Analysis",
            series="Interactive plot",
            iteration=0,
            figure=plt,
            report_interactive=True,
        )

    def start(self, model: Any, data: pd.DataFrame) -> None:

        columns_mapping = self._columns_mapping(data)
        train, test = train_test_split(
            data,
            stratify=data[TARGET.name],
            test_size=self.test_size,
            random_state=self.settings.random_seed,
        )

        samplewise_statistics = {}
        for sample_size in self.samples:
            sample = self._sample_data(
                data=train,
                sample_size=sample_size,
                stratify_column=TARGET.name,
            )
            model.fit(
                sample.drop(columns=[TARGET.name]).rename(columns=columns_mapping),
                sample[TARGET.name],
            )
            samplewise_statistics[sample_size] = balanced_accuracy_score(
                test[TARGET.name],
                model.predict(
                    test.drop(columns=[TARGET.name]).rename(columns=columns_mapping)
                ),
            )

        self._report_analysis_plot(samplewise_statistics)
