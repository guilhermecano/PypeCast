from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    """
    Calculates RMSE
    :param y_true: actual values
    :param y_pred: predicted values
    :return: RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Calculates MAE
    :param y_true: actual values
    :param y_pred: predicted values
    :return: MAE
    """
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    """
    Calculates MAPE
    :param y_true: actual values
    :param y_pred: predicted values
    :return: sMAPE
    """
    raise(NotImplementedError)

def smape(y_true, y_pred):
    """
    Calculates sMAPE
    :param y_true: actual values
    :param y_pred: predicted values
    :return: sMAPE
    """
    y_true = np.reshape(y_true, (-1,))
    y_pred = np.reshape(y_pred, (-1,))
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))).item()

def sign(y_true, y_pred):
    raise(NotImplementedError)

def tic(y_true, y_pred):
    """
    Calculates TIC metric
    :param y_true: actual values
    :param y_pred: predicted values
    :return: TIC
    """
    raise(NotImplementedError)

def mase(insample, y_true, y_pred, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return: MAsE
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_true - y_pred)) / masep

