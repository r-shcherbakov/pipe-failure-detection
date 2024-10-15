# -*- coding: utf-8 -*-
r"""Main transformers"""
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from bottleneck import (
    move_max,
    move_mean,
    move_median,
    move_min,
    move_std,
    move_sum
)
import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
import pandas as pd
import pywt

from common.constants import SECONDS_IN_HOUR
from core import BaseTransformer
from utilities.utils import (
    get_subclasses,
    reduce_memory_usage,
    get_common_timestep
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
            reduce_memory_usage(data[undefined_columns])

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


class TimeResampler(BaseTransformer):
    """Transformer for resampling data to regular time step."""

    def __init__(self, time_step: Optional[int] = None):
        self.time_step = time_step

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Resamples data to regular time step.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data with regular time step.
        """

        if not self.time_step:
            time_step = get_common_timestep(X)
        else:
            time_step = self.time_step

        return X.resample(f"{time_step}s", on="datetime").mean().reset_index()


class Aggregator(BaseTransformer):
    r"""Transformer for aggregation feature engineering"""

    def __init__(
        self,
        feature_source:str,
        feature_name: Optional[str] = None,
        filter_column: Optional[str] = None,
        filter_value: Optional[Union[int, str, pd.Categorical]] = None,
        shift_size: Optional[int] = 0,
        window: Optional[int] = None,
        min_periods: Optional[int] = None,
        agg_func: Optional[Union[str, Callable[[Any], Any], None]] = None,
        quantile: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            feature_source (str): Name of processed dataframe column.
            feature_name (Optional[str], optional): Name of output series.
                Defaults to None
            filter_column (Optional[str], optional): Name of filtering dataframe column.
                Defaults to None
            filter_value (Optional[Union[int, str, pd.Categorical]], optional): Filtering value.
                Defaults to None
            shift_size (Optional[int], optional): Number of periods to shift. Can be positive or negative.
                Defaults to 0.
            window (Optional[int], optional): Size of the moving window. Defaults to None.
            min_periods (Optional[int], optional): Minimum number of observations in window
                required to have a value. Defaults to None.
            agg_func (Optional[Union[str, Callable[[Any], Any], None]], optional], optional): Function to use
                for aggregating the data. Defaults to None.
            quantile (Optional[float], optional): Value between 0 <= q <= 1,
                the quantile(s) to compute. Defaults to None.
        """

        self.feature_source = feature_source
        self.filter_column = filter_column
        self.filter_value = filter_value
        self.agg_func = agg_func
        self.quantile = quantile
        self.window = window
        self.min_periods = min_periods
        self.shift_size = shift_size

        self.feature_name = feature_name
        if not self.feature_name:
            self.feature_name = type(self).__name__
            for el in [self.feature_source, self.filter_column, self.filter_value,
                       self.agg_func, self.quantile, self.window, self.min_periods,
                       self.shift_size]:
                self.feature_name += el if el else ""

        self.kwargs = kwargs

    def _get_aggregation(
        self,
        series: pd.Series,
        window_type: str = "expanding",
        **kwargs
    ):

        if window_type == "rolling":
            rolling_param_names = [
                "window",
                "min_periods",
                "center",
                "win_type",
                "axis",
                "closed",
            ]
            rolling_kwargs = {
                param: value
                for param, value in kwargs.items()
                if param in rolling_param_names
            }
            return series.rolling(**rolling_kwargs)
        elif window_type == "expanding":
            expanding_param_names = ["min_periods", "center", "axis"]
            expanding_kwargs = {
                param: value
                for param, value in kwargs.items()
                if param in expanding_param_names
            }
            return series.expanding(**expanding_kwargs)
        else:
            message = (
                f"{window_type} is not supported, please use 'rolling' or 'expanding'"
            )
            LOGGER.error(message)
            raise AttributeError(message)

    def _get_bottleneck_agg(
        self,
        series: pd.Series,
        window: int,
        min_periods: int,
        agg_func: Union[str, Callable[[Any], Any]],
        **kwargs,
    ) -> pd.Series:

        aggregations_dict = {
            "sum": move_sum,
            "mean": move_mean,
            "median": move_median,
            "min": move_min,
            "max": move_max,
            "std": move_std,
        }
        output = pd.Series(
            aggregations_dict[agg_func](
                series,
                window=window,
                min_count=min_periods,
                **kwargs,
            )
        )
        return output

    def _get_pandas_agg(
        self,
        series: pd.Series,
        window: int,
        min_periods: int,
        agg_func: Union[str, Callable[[Any], Any]],
        window_type: str,
        **kwargs,
    ) -> pd.Series:

        rolling = self._get_aggregation(
            series=series,
            window=window,
            min_periods=min_periods,
            window_type=window_type,
            **kwargs,
        )
        output = rolling.agg(agg_func)
        return output

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Returns series of required aggregation of input parameter.
        If input series has more than 7200 points, aggregation is calculated with bottleneck.

        Args:
            X (pd.DataFrame): Input dataframe.

        Raises:
            KeyError: Raised when feature_source column not in data columns.
            AttributeError: Raised when agg_func nor quantile provided.

        Returns:
            pd.Series: Series of required aggregation of input parameter.
        """

        if self.feature_source not in X.columns:
            raise KeyError

        if self.filter_column:
            mask = X[self.filter_column] == self.filter_value
        else:
            mask = len(X) * [True]

        window_type = "expanding" if self.window is None else "rolling"
        if self.agg_func is not None:
            if len(X.loc[mask, self.feature_source]) > SECONDS_IN_HOUR * 2:
                output = self._get_bottleneck_agg(
                    series=X.loc[mask, self.feature_source],
                    window=self.window,
                    min_periods=self.min_periods,
                    agg_func=self.agg_func,
                    **self.kwargs,
                )
            else:
                output = self._get_pandas_agg(
                    series=X.loc[mask, self.feature_source],
                    window=self.window,
                    min_periods=self.min_periods,
                    agg_func=self.agg_func,
                    window_type=window_type,
                    **self.kwargs,
                )
        elif self.quantile is not None:
            rolling = self._get_aggregation(
                series=X.loc[mask, self.feature_source],
                window=self.window,
                min_periods=self.min_periods,
                window_type=window_type,
                **self.kwargs,
            )
            output = rolling.quantile(self.quantile)
        else:
            message = (
                f"It's required to provide at least agg_func or quantile in parameters\n"
                f"for {X.name}"
            )
            LOGGER.error(message)
            raise AttributeError(message)

        X[self.feature_name] = output.reindex(X.index).shift(self.shift_size)
        return X


class Converter(BaseTransformer):
    r"""Transformer for math operations with input series"""

    def __init__(
        self,
        first_feature_source: str,
        second_feature_source: str,
        feature_name: Optional[str] = None,
        method: Optional[str] = "divide",
        **kwargs,
    ):

        self.first_feature_source = first_feature_source
        self.second_feature_source = second_feature_source
        self.method = method

        self.feature_name = feature_name
        if not self.feature_name:
            self.feature_name = type(self).__name__
            for el in [self.first_feature_source, self.second_feature_source, self.method]:
                self.feature_name += el if el else ""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns the dataframe of calculations of input parameters aggregation
            according to provided method.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with calculations of input parameters aggregation
                according to provided method.
        """
        if self.copy:
            X = X.copy()

        if not set([self.first_feature_source, self.second_feature_source]).issubset(X.columns):
            raise KeyError(f'{self.first_feature_source} and {self.second_feature_source} '
                           'are not in dataframe columns')

        first_series = X[self.first_feature_source]
        second_series = X[self.second_feature_source]
        if self.method == "divide":
            output = (first_series / second_series).astype("float32").round(3)
        if self.method == "substract":
            output = (first_series - second_series).astype("float32").round(3)
        if self.method == "normalize":
            output = ((first_series - second_series) / second_series).astype("float32").round(3)

        X[self.feature_name] = pd.Series(output).reindex(X.index)

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
