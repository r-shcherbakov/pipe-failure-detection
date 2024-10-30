# -*- coding: utf-8 -*-
from typing import List, Tuple, TYPE_CHECKING
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from pandas.util import hash_array
from sklearn.model_selection import (
    cross_validate,
    cross_val_score
)
from tqdm import tqdm

from common.features import TARGET
from common.pipeline_steps import SELECT_FEATURES
from core import BasePipelineStep
from utilities.utils import invert_dict

if TYPE_CHECKING:
    from common.pipeline_steps import PipelineStep

warnings.simplefilter(action="ignore", category=FutureWarning)


class SelectFeaturesPipelineStep(BasePipelineStep):
    def __init__(self):
        self.pipeline_step: 'PipelineStep' = SELECT_FEATURES
        super().__init__(self.pipeline_step)

        self.learning_rate = self.step_params.get('learning_rate', 0.01)
        self.n_estimators = self.step_params.get('n_estimators', 300)
        self.folds = self.step_params.get('folds', 5)
        self.metric = self.step_params.get('metric', 'roc_auc')

    def _get_columns_mapping(self, data: pd.DataFrame) -> dict:
        columns_mapping = dict(zip(
            data.columns.values,
            hash_array(data.columns.values, encoding='utf8')
        ))
        return columns_mapping

    def _get_target_correlated_features(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        correlation = data.corr().round(2)
        target_corr = correlation[TARGET.name].dropna().abs()
        cond = target_corr > 0.5
        target_correlated_features = target_corr[cond].index[:-1].tolist()
        self.task.upload_artifact(
            name='target correlated features',
            artifact_object=target_correlated_features
        )
        return target_correlated_features

    def _get_less_correlated_features(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        correlation = data.drop(columns=[TARGET.name]).corr()
        corr_stat = {}
        for col in correlation.columns:
            corr_stat[col] = np.abs(correlation[col]).sum(axis=0)

        corr_stat = pd.Series(corr_stat)
        cond = corr_stat <= corr_stat.quantile(0.05)
        less_correlated_features = corr_stat[cond].index.tolist()
        self.task.upload_artifact(
            name='less correlated features',
            artifact_object=less_correlated_features
        )
        return less_correlated_features

    def _get_all_features_stat(
        self,
        data: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[float, List[str]]:
        model = LGBMClassifier(
            random_state=self.settings.random_seed,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbose=-1,
        )
        output = cross_validate(
            estimator=model,
            X=data,
            y=labels,
            cv=self.folds,
            scoring=self.metric,
            return_estimator=True
        )

        feature_importances = []
        for estimator in output['estimator']:
            feature_importances.append(estimator.feature_importances_)
        feature_importances = pd.DataFrame(
            data=np.array(feature_importances).T,
            columns=[
                f'importance {str(idx)} fold'
                for idx in range(len(feature_importances))],
            index=data.columns,
        )
        feature_importances['mean_importance'] = feature_importances.mean(axis=1)
        features = feature_importances['mean_importance'] \
            .sort_values(ascending=True).index.tolist()
        score = output['test_score'].mean()

        feature_importances = feature_importances.reset_index(names=['feature'])
        feature_importances['feature'] = feature_importances['feature'] \
            .map(invert_dict(self.columns_mapping))
        self.task.upload_artifact(
            name='all features importances',
            artifact_object=feature_importances
        )

        return score, features

    def _get_most_significant_features(
        self,
        data: pd.DataFrame
    ) -> List[str]:

        self.columns_mapping = self._get_columns_mapping(data)
        X = data.drop(columns=[TARGET.name]) \
            .rename(columns=self.columns_mapping)
        y = data[TARGET.name]
        score_all_features, sorted_features = self._get_all_features_stat(X, y)

        metric_diff_threshold = self.step_params.get('metric_diff_threshold', 0.001)
        features_to_remove = []
        feature_cv_scores_mean = []
        feature_metric_diff = []
        for feature in tqdm(sorted_features, total=len(sorted_features)):
            model = LGBMClassifier(
                random_state=self.settings.random_seed,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                verbose=-1,
            )
            cv_score = cross_val_score(
                estimator=model,
                X=X.drop(columns=features_to_remove + [feature]),
                y=y,
                scoring=self.metric,
                cv=self.folds,
            )
            cv_score_mean = cv_score.mean()
            feature_cv_scores_mean.append(cv_score_mean)

            score_diff = score_all_features - cv_score_mean
            feature_metric_diff.append(score_diff)

            if score_diff <= metric_diff_threshold:
                score_all_features = cv_score_mean
                features_to_remove.append(feature)

        inverted_columns_mapping = invert_dict(self.columns_mapping)
        debug_data = pd.DataFrame({
            'feature': [inverted_columns_mapping[feature] for feature in sorted_features],
            'mean_feature_metric': feature_cv_scores_mean,
            'feature_metric_diff': feature_metric_diff,
        })
        self.task.upload_artifact(
            name='features score gain',
            artifact_object=debug_data
        )

        features_to_keep = [inverted_columns_mapping[x] for x in sorted_features \
            if x not in features_to_remove]

        return features_to_keep

    def start(self, data: pd.DataFrame) -> List[str]:
        target_correlated_features = self._get_target_correlated_features(data)
        less_correlated_features = self._get_less_correlated_features(data)
        filtered_features = target_correlated_features + less_correlated_features + [TARGET.name]
        data = data[filtered_features]

        selected_features = self._get_most_significant_features(data)
        return selected_features
