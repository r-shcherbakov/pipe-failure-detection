# -*- coding: utf-8 -*-
r"""Main transformers"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from bottleneck import move_max, move_mean, move_median, move_min, move_std, move_sum
import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
import pandas as pd
from pyod.models.iforest import IForest
import pywt
from sklearn.pipeline import Pipeline

from common.config import ACCEPTED_BOUNDARIES, FILLNA_CONFIG
from common.constants import SECONDS_IN_HOUR, SECONDS_IN_MINUTE
from core.transformer import BaseTransformer
from utilities.utils import get_subclasses, convert_columns_type, get_common_timestep

LOGGER = logging.getLogger(__name__)


class DuplicatedColumnsTransformer(BaseTransformer):
    """Drops duplicated columns and leaves the most filled."""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._drop_duplicated_columns(data)
        return data
    
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
    r"""Transformer for converting column values type according to config."""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = convert_columns_type(data)
        return data
    
    
class ClipTransformer(BaseTransformer):
    """Transformer for removing data outliers with min and max accepted values"""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes data outliers according to config.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data with clipped values exceeding the boundaries.
        """
        data = self._outlier_correction(data, boundaries=ACCEPTED_BOUNDARIES)
        LOGGER.debug(
            f"ClipTransformer removes outliers, results shape is {data.shape}"
        )
        return data
    
    def _outlier_correction(self, data: pd.DataFrame, boundaries: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Assigns values outside boundary to boundary values.

        Args:
            data (pd.DataFrame): Input data.
            boundaries (Dict[str, Dict[str, float]]): Config with boundaries for columns.

        Returns:
            pd.DataFrame: Input data with clipped values exceeding the boundaries.
        """

        for column in boundaries:
            if column in data.columns:
                data[column] = data[column].clip(
                    lower=boundaries[column]["min"], upper=boundaries[column]["max"]
                )
        return data


class InfValuesTransformer(BaseTransformer):
    """Transformer for replacing infinite values with nans."""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replaces infinite values with nans.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data without infinite values.
        """

        LOGGER.debug(f"InfValuesTransformer found {np.isinf(data).values.sum()} infinite values")
        data = data.replace([np.inf, -np.inf], np.nan)
        return data
    
    
class FillNanTransformer(BaseTransformer):
    """Transformer for replacing missing values with values according to config."""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replaces missing values with values according to config.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data without missing values.
        """

        LOGGER.debug(f"FillNanTransformer found {data.isna().sum().sum()} NaN values")
        data = self._fill_missing_values(data, config=FILLNA_CONFIG)
        return data
    
    def _fill_missing_values(self, data: pd.DataFrame, config: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Replaces missing values with values according to config.

        Args:
            data (pd.DataFrame): Input data.
            config (Dict[str, Dict[str, float]]): Config with fillna method
                parameters for specified columns.

        Returns:
            pd.DataFrame: Input data without missing values.
        """

        specified_features = list(
            np.intersect1d(data.columns, list(config.keys()))
            )
        nonspecified_features = list(
            np.setdiff1d(data.columns, list(config.keys()))
            )

        for column in specified_features:
            if column in data.columns:
                data[column] = data[column].fillna(
                    value=config[column]["value"],
                    method=config[column]["method"],
                    limit=config[column]["limit"],
                )
        for column in nonspecified_features:
            data[column] = data[column].ffill(limit=SECONDS_IN_MINUTE*10)
            
        return data


class TimeResampler(BaseTransformer):
    """Transformer for resampling data to regular time step."""

    def __init__(self, time_step: Optional[int] = None):
        self.time_step = time_step

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resamples data to regular time step.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Input data with regular time step.
        """
        
        if not self.time_step:
            time_step = get_common_timestep(data)
        else:
            time_step = self.time_step

        return data.resample(f"{time_step}s", on="datetime").mean().reset_index()
    
    
class OutlierImputer(BaseTransformer):
    r"""Transformer for impiting outliers with unsupervized machine learning model"""

    def transform(self, series: pd.Series) -> np.ndarray:
        if not series.empty:
            clf = IForest()
            clf.fit(series)
            
            return self._np_ffill(np.where(clf.labels_, np.nan, series.reshape(-1)), axis=0)
        else:
            return series
        
    def _np_ffill(self, arr: np.ndarray, axis: int):
        idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
        idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
        np.maximum.accumulate(idx, axis=axis, out=idx)
        slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
            for dim in range(len(arr.shape))])]
            for i, k in enumerate(arr.shape)]
        slc[axis] = idx
        return arr[tuple(slc)]


class Aggregator(BaseTransformer):
    r"""Transformer for aggregation feature engineering"""

    def __init__(
        self,
        shift_size: int = 0,
        window: Optional[int] = None,
        min_periods: Optional[int] = None,
        mask: Optional[List[bool]] = None,
        agg_func: Union[str, Callable[[Any], Any], None] = None,
        quantile: Optional[float] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)        
        self.shift_size = shift_size
        self.window = window
        self.min_periods = min_periods
        self.mask = mask
        self.agg_func = agg_func
        self.quantile = quantile

    def _get_aggregation(
        self, series: pd.Series, window_type: str = "expanding", **kwargs
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
        
    def transform(self, data: pd.Series) -> pd.Series:
        """Returns the series of required aggregation of input parameter.
        If input series has more than 7200 points, aggregation is calculated with bottleneck.

        Args:
            data (pd.Series): Input series.
            shift_size (int): Number of periods to shift. Can be positive or negative.
            window (Optional[int], optional): Size of the moving window. Defaults to None.
            min_periods (Optional[int], optional): Minimum number of observations in window required to have a value. Defaults to None.
            mask (Optional[List[bool]], optional): _description_. Defaults to None.
            agg_func (Union[str, Callable[[Any], Any], None], optional): Function to use for aggregating the data. Defaults to None.
            quantile (Optional[float], optional): Value between 0 <= q <= 1, the quantile(s) to compute. Defaults to None.

        Raises:
            AttributeError: Raised when agg_func nor quantile provided.

        Returns:
            pd.Series: Series of required aggregation of input parameter.
        """

        window_type = "expanding" if self.window is None else "rolling"
        mask = data.size * [True] if mask is None else mask

        if self.agg_func is not None:
            if len(data[mask]) > SECONDS_IN_HOUR * 2:
                output = self._get_bottleneck_agg(
                    series=data[mask],
                    window=self.window,
                    min_periods=self.min_periods,
                    agg_func=self.agg_func,
                    **self.kwargs,
                )
            else:
                output = self._get_pandas_agg(
                    series=data[mask],
                    window=self.window,
                    min_periods=self.min_periods,
                    agg_func=self.agg_func,
                    window_type=window_type,
                    **self.kwargs,
                )
        elif self.quantile is not None:
            rolling = self._get_aggregation(
                series=data[mask],
                window=self.window,
                min_periods=self.min_periods,
                window_type=window_type,
                **self.kwargs,
            )
            output = rolling.quantile(self.quantile)
        else:
            message = (
                f"It's required to provide at least agg_func or quantile in parameters\n"
                f"for {data.name}"
            )
            LOGGER.error(message)
            raise AttributeError(message)
        output.name = None
        output = output.reindex(data.index).shift(self.shift_size)
        return output
    
    
class Converter(BaseTransformer):
    r"""Transformer for math operations with input series"""
    
    def __init__(
        self,
        method: str = None,
        first_shift_size: int = 0,
        second_shift_size: int = 0,
        first_window: Optional[int] = None,
        second_window: Optional[int] = None,
        first_min_periods: Optional[int] = None,
        second_min_periods: Optional[int] = None,
        first_mask: Optional[List[bool]] = None,
        second_mask: Optional[List[bool]] = None,
        first_agg_func: Union[str, Callable[[Any], Any], None] = None,
        second_agg_func: Union[str, Callable[[Any], Any], None] = None,
        first_quantile: Optional[float] = None,
        second_quantile: Optional[float] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.method = "divide" if method is None else method
        self.first_mask = first_mask
        self.first_window = first_window
        self.first_shift_size = first_shift_size
        self.first_min_periods = first_min_periods
        self.first_agg_func = first_agg_func
        self.first_quantile = first_quantile
        self.second_mask = second_mask
        self.second_window = second_window
        self.second_shift_size = second_shift_size
        self.second_min_periods = second_min_periods
        self.second_agg_func = second_agg_func
        self.second_quantile = second_quantile
        
    def transform(self, data: pd.Series) -> pd.Series:
        """Returns the series of calculations of input parameters aggregation
            according to provided method.

        Args:
            data (pd.Series): Input series.

        Returns:
            pd.Series: Series of calculations of input parameters aggregation
            according to provided method.
        """

        first_parameters = {
            key.replace("first_", ""): val
            for key, val in self.__dict__.items()
            if "first_" in key
        }
        first_series = Aggregator(**first_parameters).fit_transform(data).ffill()
        
        second_parameters = {
            key.replace("second_", ""): val
            for key, val in self.__dict__.items()
            if "second_" in key
        }
        second_series = Aggregator(**second_parameters).fit_transform(data).ffill()
        
        if self.method == "divide":
            output = (first_series / second_series).astype("float32").round(3)
        if self.method == "substract":
            output = (first_series - second_series).astype("float32").round(3)
        if self.method == "normalize":
            output = ((first_series - second_series) / second_series).astype("float32").round(3)

        return output
    

class Shifter(Aggregator, BaseTransformer):
    r"""Transformer for shifted feature engineering"""

    def __init__(
        self,
        shift_size: int,
        window: Optional[int] = None,
        min_periods: Optional[int] = None,
        mask: Optional[List[bool]] = None,
        agg_func: Union[str, Callable[[Any], Any], None] = None,
        quantile: Optional[float] = None,
        **kwargs,
    ):

        self.mask = mask
        self.window = window
        self.shift_size = shift_size
        self.min_periods = min_periods
        self.agg_func = agg_func
        self.quantile = quantile

    def transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Returns the series of calculations of input parameters shifted
            aggregation according to provided method.

        Args:
            series (pd.Series): Input series.

        Returns:
            pd.Series: Series of calculations of input parameters shifted
                aggregation according to provided method.
        """

        parameters = self.__dict__
        output = self.transform_one_series(series, **parameters).ffill()
        output.name = None
        output = (
            output.reindex(series.index)
            .shift(self.shift_size)
            .astype("float32")
            .round(3)
        )
        return output


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

    def __init__(self, pos_value=0.01):
        self.pos_value = pos_value
    
    def transform(self, series: pd.Series) -> pd.Series:
        series[series <= 0] = self.pos_value
        return series


ALL_TRANSFORMERS = {
    transformer.__name__: transformer
    for transformer in get_subclasses(BaseTransformer)
}
