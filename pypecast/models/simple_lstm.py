from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from copy import copy

from pypecast.models.model import Model

class Simple_LSTM(Model):
    '''Class for defining and training a common LSTM network for time series forecasting'''
    def __init__(self, 
                n_lag, 
                n_seq,
                n_units= None,
                hidden = None,
                dropout = None, 
                hidden_activation = 'relu'):

        super(Simple_LSTM, self).__init__(
            n_lag = n_lag,
            n_seq=n_seq
        )
        
        #Architecture
        self._n_units = n_units
        self._hidden = hidden
        self._dropout = dropout
        self._activation = hidden_activation


    def _design_network(self, out_shape):
        # design network
        model = Sequential()
        model.add(LSTM(self._n_units))

         #Add first layer dropout
        if self._dropout is not None:
            if isinstance(self._dropout,(list,)):
                model.add(Dropout(self._dropout[0]))
            else:
                model.add(Dropout(self._dropout))
        #
        if self._hidden is not None:
            #verify if it is list
            assert isinstance(self._hidden, (list,)), 'Hidden is not a list()'
            #check if dropout was defined in a correspondent manner
            if len(self._hidden) == len(self._dropout)-1:
                dropout_list = copy(self._dropout)
                dropout_list.pop(0)
                for n, drop in zip(self._hidden,dropout_list):
                    model.add(Dense(n,activation=self._activation))
                    model.add(Dropout(drop))
            #or if it was not defined for the hidden layers
            elif self._dropout is None:
                for n in self._hidden:
                    model.add(Dense(n,activation=self._activation))    

        model.add(Dense(out_shape))
        self._model = model

    def fit(self, train,n_batch = 1, n_epoch = 1000, loss = 'mse', opt = 'adam', early_stopping = False):
        X, y = train[:, 0:self._n_lag], train[:, self._n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1]) #n_series,  n_timesteps, n_features
        
        # design network
        self._design_network(y.shape[1])
        #compile model
        self._model.compile(loss=loss, optimizer=opt)

        #check early stopping
        if isinstance(early_stopping, (bool,)):
            #default es
            if early_stopping:
                es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None)
        else:
            es = early_stopping
        # fit network
        self._model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2, shuffle=False, validation_split=0.2, callbacks=[es])
        
        print('-'*60)
        print('Model trained')

    # def forecast_series(self, test, scaler, orig_series):
    #     assert self._model is not None, "Model must be trained first"

    #     forecasts = list()
    #     for i in range(len(test)):
    #         X, y = test[i, 0:self._n_lag], test[i, self._n_lag:]
    #         # make forecast
    #         forecast = self._forecast_model(X)
    #         # store the forecast
    #         forecasts.append(forecast)
        
    #     #inverse_transform
    #     forecasts = self._inverse_transform(orig_series,forecasts,scaler,test.shape[0])

    #     self._forecasts = forecasts
    #     return forecasts

        