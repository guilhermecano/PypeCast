from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from pypecast.metrics.metrics import *

class Model(object):
    '''Base class for models'''
    
    def __init__(self, 
                n_lag, 
                n_seq):

        self._model = None
        self._forecasts = None

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

    def forecast_series(self, test, scaler, orig_series):
        assert self._model is not None, "Model must be trained first"

        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:self._n_lag], test[i, self._n_lag:]
            # make forecast
            forecast = self._forecast_model(X)
            # store the forecast
            forecasts.append(forecast)

        #inverse_transform
        forecasts = self._inverse_transform(orig_series,forecasts,scaler,test.shape[0])
        self._forecasts = forecasts
        return forecasts

    def get_forecast(self):
        return self._forecasts

        # invert differenced forecast
    def _inverse_difference(self,last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i-1])
        return inverted

    # inverse data transform on forecasts
    def _inverse_transform(self, series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = np.array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            if scaler[0] is not None:
                forecast = scaler[0].inverse_transform(forecast)
                forecast = forecast[0, :]
            
            if scaler[1]:
                # invert differencing
                index = len(series) - n_test + i - 1
                last_ob = series.values[index]
                forecast = self._inverse_difference(last_ob, forecast)

            inverted.append(forecast)
        return inverted
    
    def evaluate_forecast(self, test, forecasts):
        print('-'*20 + 'Forecast evaluation' + '-'*20)
        for i in range(self._n_seq):
            actual = [row[i] for row in test]        
            #print(np.array(actual))
            predicted = [forecast[i] for forecast in self._forecasts]
            #RMSE
            print('t+%d RMSE: %f' % ((i+1), rmse(actual, predicted)))
            #MAE
            print('t+%d MAE: %f' % ((i+1), mae(actual, predicted)))
            #sMAPE
            print('t+%d sMAPE: %f' % ((i+1), smape(actual, predicted)))

    def plot_forecasts(self, series, forecasts, test):
        n_test = test.shape[0]
        sns.set()
        # plot the entire dataset in blue
        warnings.filterwarnings("ignore")
        plt.figure(0,figsize=[12,6])
        plt.plot(series.values, label='True time-series')
        # plot the forecasts in red
        for i in range(len(forecasts)):
            off_s = len(series) - n_test + i
            off_e = off_s + len(forecasts[i])
            xaxis = [x for x in range(off_s, off_e)]
            plt.plot(xaxis, forecasts[i], color='red',label='Forecasted time-series')
        # show the plot
        plt.show()
        sns.reset_defaults()