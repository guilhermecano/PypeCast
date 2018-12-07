from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from pypecast.features import BuildFeatures

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BuildFeaturesSupervised(BuildFeatures):
    '''Class for building features from a single series into a supervised learning data'''
    def __init__(self, n_lag, n_seq, test_split = 0.25, n_test=None, difference= False, use_log = False, scaler_type = None):
        
        super(BuildFeaturesSupervised, self).__init__(
            n_lag = n_lag,
            n_seq=n_seq,
            difference = difference,
            use_log = use_log,
            scaler_type = scaler_type
        )

        if n_test is None:
            self._use_split = True
        else:
            self._use_split = False

        self._scaler = None

        self._n_test = n_test
        self._test_split = test_split
    
    def _series_to_supervised(self, data, dropnan=True):
        if type(data) is list:
            n_vars = 1
        else:
            try: 
                n_vars = data.shape[1] 
            except: 
                n_vars = 1
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self._n_lag, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self._n_seq):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # transform series into train and test sets for supervised learning
    def transform_data(self, series):
        # extract raw values
        raw_values = series.values

        ## apply log into the series
        # todo

        # transform data to be stationary in difference is set
        if self._difference:    
            diff_series = self._make_difference(raw_values, 1)
        else:
            diff_series = series
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)

        #apply scaling or not
        if self._scaler_type is not None:
            # rescale values to -1, 1
            if self._scaler_type == 'norm':
                self._scaler = MinMaxScaler(feature_range=(-1, 1))
            elif self._scaler == 'std':
                self._scaler = StandardScaler()
            else:
                self._scaler = self._scaler_type
            scaled_values = self._scaler.fit_transform(diff_values)
        else:
            scaled_values = diff_values
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        
        # transform into supervised learning problem X, y
        supervised = self._series_to_supervised(scaled_values)
        supervised_values = supervised.values
        # split into train and test sets
        if self._use_split == False:
            train, test = supervised_values[0:-self._n_test], supervised_values[-self._n_test:]
        else:
            n_test = int(self._test_split*supervised_values.shape[0])
            train, test = supervised_values[:-n_test], supervised_values[-n_test:]
        return (self._scaler,self._difference), train, test



        

