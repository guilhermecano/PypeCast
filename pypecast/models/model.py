from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

class Model(object):
    '''Base class for models'''
    
    def __init__(self, 
                n_lag, 
                n_seq):

        self._model = None
        self._forecast = None
    
    def fit(self):
        raise(NotImplementedError)

    def forecast_series(self):
        raise(NotImplementedError)

    def get_forecast(self):
        return self._forecast