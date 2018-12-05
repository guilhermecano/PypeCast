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

        #Must have variables
        self._n_lag = n_lag
        self._n_seq = n_seq
    
    def summary(self):
        if self._model is not None:
            self._model.show()
        else:
            print('The model was not defined yet. Please use the fit() method first.')

    def fit(self):
        raise(NotImplementedError)

    def _forecast_model(self, X):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = self._model.predict(X)
        # convert to array
        return [x for x in forecast[0, :]]

    def forecast_series(self):
        raise(NotImplementedError)

    def get_forecast(self):
        return self._forecast