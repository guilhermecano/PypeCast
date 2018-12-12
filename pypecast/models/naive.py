from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from pypecast.models import Model

class Naive_model(object):
    def __init__(self, 
                n_lag, 
                n_seq):
    
        self._n_lag = n_lag
        self._n_seq = n_seq
        self._forecasts = None

    def _fit_naive(self, last_ob, n_seq):
        return [last_ob for i in range(self._n_seq)]

    def forecast_series(self, test, scaler, orig_series):
        '''Forecasts naive_model. Not compatible with transformations yet, 
        so it is recommended to don'n scale the data for this model'''
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:self._n_lag], test[i, self._n_lag:]
            # make forecast
            forecast = self._fit_naive(X[-1], self._n_seq)
            # store the forecast
            forecasts.append(forecast)
        
        self._forecasts = forecasts
        return forecasts