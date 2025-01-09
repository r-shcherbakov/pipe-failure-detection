# -*- coding: utf-8 -*-
from typing import Tuple, List
import warnings

from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.common.enums import DefectType
from src.common.exceptions import SplitDataError
from src.common.features import TARGET
from src.common.pipeline_steps import SPLIT_DATASET
from src.core import BasePipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


class SplitDatasetPipelineStep(BasePipelineStep):
    def __init__(self):
        super().__init__(SPLIT_DATASET)

    def _check_dataset_drift(
        self,
        data: pd.DataFrame,
        test_index: List[int]
    ) -> bool:
        drift_report = Report(metrics=[
            DatasetDriftMetric(
                    num_stattest=self.step_params.get('num_stattest', 'ks'),
                    num_stattest_threshold=self.step_params.get('num_stattest_threshold', 0.5),
                    drift_share=self.step_params.get('drift_share', 0.5),
            ),
        ])

        drift_report.run(
            reference_data=data.iloc[~test_index].reset_index(drop=True),
            current_data=data.iloc[test_index].reset_index(drop=True),
        )

        dataset_drift = drift_report.as_dict()['metrics'][0]['result']['dataset_drift']
        return dataset_drift

    def _get_test_objects(self, data: pd.DataFrame) -> List[str]:
        test_size = self.step_params.get('test_size', 0)
        test_size = max(0, min(test_size, 0.5))
        if test_size == 0:
            return []
        else:
            n_splits = int(1 / test_size) * 2
            splitter = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=self.settings.random_seed,
            )

            X = data.drop(columns=[TARGET.name])
            y = data[TARGET.name]
            for _, test_index in splitter.split(X, y):
                data_drift = self._check_dataset_drift(X, test_index)
                if not data_drift:
                    return X.iloc[test_index].index.tolist()

            raise SplitDataError


    def start(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Pop unlabeled data
        unlabeled = data[data[TARGET.name] == DefectType.UNDEFINED.value].copy()
        data = data.drop(index=unlabeled.index)

        test_ids = self._get_test_objects(data)
        if test_ids:
            self.task.upload_artifact(
                name='test objects',
                artifact_object={"test_objects": test_ids},
            )
            test = data[data.index.isin(test_ids)].copy()
            train = data[~data.index.isin(test_ids)].copy()
        else:
            test = pd.DataFrame(columns=data.columns)
            train = data

        return train, test, unlabeled
