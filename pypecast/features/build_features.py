from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BuildFeatures(object):
    '''Base class for building features from a time-series'''
    def __init__(self, n_lag, n_seq, difference= False, use_log = False, scaler_type = None):
        self._n_lag = n_lag
        self._n_seq = n_seq

        self._difference = difference
        self._use_log = use_log
        self._scaler_type = scaler_type

    # create a differenced series
    def _make_difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)



        

