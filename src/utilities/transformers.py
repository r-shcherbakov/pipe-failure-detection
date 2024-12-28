# -*- coding: utf-8 -*-
r"""Main transformers"""
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
import pandas as pd
import pywt

from core import BaseTransformer
from utilities.utils import (
    get_subclasses,
    reduce_memory_usage,
)

LOGGER = logging.getLogger(__name__)


class DuplicatedColumnsTransformer(BaseTransformer):
    """Drops duplicated columns and leaves the most filled."""
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._drop_duplicated_columns(X)
        return X

    def _drop_duplicated_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicated columns and leaves the most filled.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input dataframe without duplicated columns and empty columns.
        """

        ldf = data.copy()
        duplicated_columns = pd.Series(ldf.columns).value_counts()[
            pd.Series(ldf.columns).value_counts() > 1
        ]
        for col in duplicated_columns.index:
            temp = ldf.pop(col)
            LOGGER.warning(f"New case of duplicate columns: {col}. Will take the most filled one")
            ldf.loc[:, col] = temp.iloc[:, np.argmax(temp.notnull().mean().values)]
        return ldf


class ColumnsTypeTransformer(BaseTransformer):
    r"""Transformer for converting column type according to config."""
    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._convert_columns_type(X)
        return X

    def _convert_columns_type(self, data: pd.DataFrame) -> pd.DataFrame:
        """Converts columns type according to config.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data with converted columns type.
        """
        columns = data.columns.tolist()
        if self.config:
            undefined_columns = np.setdiff1d(
                np.unique(columns),
                list(self.config.keys()),
            )
            specified_features = np.intersect1d(
                np.unique(columns),
                list(self.config.keys()),
            )
            data.astype({
                feature:config for feature, config in self.config.items()
                if feature in specified_features
            })
        else:
            undefined_columns = np.unique(columns)

        if len(undefined_columns) > 0:
            data[undefined_columns] = reduce_memory_usage(data[undefined_columns])

        return data


class ClipTransformer(BaseTransformer):
    """Transformer for removing data outliers with min and max accepted values"""
    def __init__(self, config: Optional[Dict[str, Dict[str, float]]] = None):
        self.config = config

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes data outliers according to config.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data with clipped values exceeding the boundaries.
        """
        if self.config:
            specified_features = np.intersect1d(
                np.unique(X.columns),
                list(self.config.keys()),
            )
            for column in specified_features:
                X[column] = X[column].clip(
                    lower=self.config[column]["lower"],
                    upper=self.config[column]["upper"],
                )

            LOGGER.debug(
                f"ClipTransformer removes outliers, results shape is {X.shape}"
            )
        else:
            LOGGER.debug(
                "Config is missing, skipping clipping outliers"
            )

        return X


class InfValuesTransformer(BaseTransformer):
    """Transformer for replacing infinite values with nans."""
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces infinite values with nans.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data without infinite values.
        """
        count = np.isinf(X.select_dtypes(exclude=['category', 'object'])).values.sum()
        LOGGER.debug(
            f"ReplaceInfValues found {count} "
            "infinite values"
        )
        if count > 0:
            X = X.replace([np.inf, -np.inf], np.nan)

        return X


class FillNanTransformer(BaseTransformer):
    """Transformer for replacing missing values with values according to config."""
    def __init__(
        self,
        config: Optional[Dict[str, Dict[str, Union[Union[float, int], Optional[str]]]]] = None
    ):
        self.config = config

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces missing values with values according to config.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data without missing values.
        """

        LOGGER.debug(f"FillNanTransformer found {X.isna().sum().sum()} NaN values")

        if self.config:
            specified_features = np.intersect1d(
                np.unique(X.columns),
                list(self.config.keys()),
            )
            for column in specified_features:
                method = self.config[column]["method"]
                value = self.config[column]["method"]
                if value:
                    X[column] = X[column].fillna(
                        value=value,
                        limit=self.config[column]["limit"],
                    )
                elif method == "bfill":
                    X[column] = X[column].bfill(
                        limit=self.config[column]["limit"],
                    )
                elif method == "ffill":
                    X[column] = X[column].ffill(
                        limit=self.config[column]["limit"],
                    )
                else:
                    LOGGER.debug(
                        f"FillNanTransformer: Fillna method of {column} column is not supported, "
                        "skipping"
                    )
        else:
            LOGGER.debug(
                "Config is missing, skipping filling missing values"
            )

        return X


class FourierTransformer(BaseTransformer):
    r"""Transformer for denoising input series with FFT approach"""

    def __init__(self, threshold: float = 1e8, **kwargs):

        super().__init__(**kwargs)
        self.threshold = threshold

    def transform(self, series: pd.Series) -> np.ndarray:
        """Returns denoised array.

        Args:
            series (pd.Series): Input series.

        Returns:
            np.ndarray: Denoised with FFT array.
        """

        series = series.to_numpy()
        fourier = rfft(series)
        frequencies = rfftfreq(series.size, d=1e-5)
        fourier[frequencies > self.threshold] = 0
        output = irfft(fourier).astype("float32")
        return output


class WaveletTransformer(BaseTransformer):
    r"""Transformer for denoising input series with wavelet families"""

    def __init__(self, wavelet: str = "db4", level: int = 1, **kwargs):

        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.level = level

    def madev(self, d, axis: Union[int, Tuple[int, int], None] = None):
        """Returns mean absolute deviation of a signal"""
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def transform(self, series: pd.Series) -> np.ndarray:
        """Returns denoised array.

        Args:
            series (pd.Series): Input series.

        Returns:
            np.ndarray: Denoised array with wavelet families.
        """

        series = series.to_numpy()
        coeff = pywt.wavedec(series, self.wavelet, mode="per")
        sigma = (1 / 0.6745) * self.madev(coeff[-self.level])
        u_threshold = sigma * np.sqrt(2 * np.log(len(series)))
        coeff[1:] = (
            pywt.threshold(i, value=u_threshold, mode="hard") for i in coeff[1:]
        )
        output = pywt.waverec(coeff, self.wavelet, mode="per").astype("float32")
        return output


class PositiveReplacer(BaseTransformer):
    r"""Transformer for replacing negative values of input series with specified positive value"""

    def __init__(self, pos_value = 0.01):
        self.pos_value = pos_value
        super().__init__()

    def transform(self, X: pd.Series) -> pd.Series:
        if self.copy:
            X = X.copy()

        numeric_features = X.select_dtypes(include='number').columns
        mask = X[numeric_features] <= 0
        X[mask] = np.float32(self.pos_value)
        return X


ALL_TRANSFORMERS = {
    transformer.__name__: transformer
    for transformer in get_subclasses(BaseTransformer)
}
