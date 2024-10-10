from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Metric(ABC):
    """
    Abstract method for computing metric
    Args:
        name: metric name
        mean_function: function for computing mean value
    """
    def __init__(self, name: str, mean_function: Callable = np.mean, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._mean_function = mean_function
        self._results = []

    def __call__(self, *args, **kwargs) -> float:
        """
        Compute metric for given values.
        Args:
            *args:
            **kwargs:

        Returns:
            metric value
        """
        res = self._compute(*args, **kwargs)
        self._results.append(res)
        return res

    def get_results(self) -> float:
        """
        Get mean value of all computed results.
        Returns:
            mean value of all computed results

        """
        return self._mean_function(self._results)

    def reset(self):
        """
        Reset metric
        """
        self._results = []

    @abstractmethod
    def _compute(self, *args, **kwargs) -> float:
        """
        Compute metric for given values
        Args:
            *args:
            **kwargs:

        Returns:
            metric value
        """
        raise NotImplementedError


class RMSE(Metric):
    """
    Root Mean Squared Error (RMSE) metric for regression problems.
    """
    def __init__(self, **kwargs):
        super().__init__('RMSE', np.median, **kwargs)

    def _compute(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


class R2(Metric):
    """
    Coefficient of determination (R2) metric for regression problems.
    """
    def __init__(self, **kwargs):
        super().__init__('R2', np.median, **kwargs)

    def _compute(self, y_true, y_pred):
        return r2_score(y_true, y_pred)


class MAE(Metric):
    """
    Mean Absolute Error (MAE) metric for regression problems.
    """
    def __init__(self, **kwargs):
        super().__init__('MAE', np.median, **kwargs)

    def _compute(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


class MSE(Metric):
    """
    Mean Squared Error (MSE) metric for regression problems.
    """
    def __init__(self, **kwargs):
        super().__init__('MSE', np.median, **kwargs)

    def _compute(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=True)


class MAPE(Metric):
    """
    Mean Absolute Percentage Error (MAPE) metric for regression problems.
    """
    def __init__(self, **kwargs):
        super().__init__('MAPE', np.median, **kwargs)

    def _compute(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100


class MEAN(Metric):
    """
    Mean of predicted values.
    """
    def __init__(self, **kwargs):
        super().__init__('mean', **kwargs)

    def _compute(self, y_true, y_pred):
        return np.mean(y_pred)


class STD(Metric):
    """
    Standard deviation of predicted values.
    """
    def __init__(self, **kwargs):
        super().__init__('std', **kwargs)

    def _compute(self, y_true, y_pred):
        return np.std(y_pred)
