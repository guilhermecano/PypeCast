from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Dropout, LSTM, Input
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
        assert n_lag > 0 and n_seq >0, 'n_seq and n_lag cant be null or negative'
        
        #Auxiliar Vars
        self._model = None
        self._forecasts = None
        self._actual = None

        self._test_idx = None
        self._idxs = None

        #Must define variables
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
        
    def _design_network(self, out_shape):
        raise(NotImplementedError)

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
        forecasts = self._inverse_transform(orig_series,forecasts,scaler,test.shape[0]+2)
        self._forecasts = forecasts
        #Actual values
        actual = [row[self._n_lag:] for row in test]
        self._actual = self._inverse_transform(orig_series, actual, scaler, test.shape[0]+2)

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
    
    def evaluate_forecast(self, save_report = False, filename = '../reports/evaluation.xlsx', return_dicts = False):
        print('-'*20 + 'Forecast evaluation' + '-'*20)
        print('')
        steps_metrics = dict()
        instant_metrics = {'RMSE':[], 'MAE':[], 'MAPE':[], 'sMAPE': []}
        
        # Metrics for each timestep in future
        for i in range(self._n_seq):
            print('Instant t+{}'.format(i+1))
            actual = [row[i] for row in self._actual]        
            #print(np.array(actual))
            predicted = [forecast[i] for forecast in self._forecasts]
            #RMSE
            m1 = rmse(actual, predicted)
            print('t+%d RMSE: %f' % ((i+1), m1))
            #MAE
            m2 = mae(actual, predicted)
            print('t+%d MAE: %f' % ((i+1), m2))
            #MAPE
            m3 = mape(actual, predicted)
            print('t+%d MAPE: %f' % ((i+1), m3))
            #sMAPE
            m4 = smape(actual, predicted)
            print('t+%d sMAPE: %f' % ((i+1),m4))
            #MASE
            #print('t+%d MASE: %f' % ((i+1), mase(actual, predicted)))

            steps_metrics[(i+1)] = [m1,m2,m3,m4]
            print('-'*60)

        # Metrics for each instant in time-series 
        for i in range(len(self._actual)):
            #RMSE
            m1 = rmse(self._actual[i], self._forecasts[i])
            #MAE
            m2 = mae(self._actual[i], self._forecasts[i])
            #MAPE
            m3 = mape(self._actual[i], self._forecasts[i])
            #sMAPE
            m4 = smape(self._actual[i], self._forecasts[i])
            
            instant_metrics['RMSE'].append(m1)
            instant_metrics['MAE'].append(m2)
            instant_metrics['MAPE'].append(m3)
            instant_metrics['sMAPE'].append(m4)

        if save_report:
            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            df1 = pd.DataFrame(steps_metrics, index=['RMSE','MAE','MAPE','sMAPE'])
            df2 = pd.DataFrame(instant_metrics)

            # Write each dataframe to a different worksheet.
            df1.to_excel(writer, sheet_name='Metrics by forecasted step')
            df2.to_excel(writer, sheet_name='Metrics at each index')
            #df3.to_excel(writer, sheet_name='Nayve forecast')

            writer.save()
        
        if return_dicts:
            return steps_metrics, instant_metrics


    def plot_forecasts(self, series, forecasts, test):
        n_test = test.shape[0]+2
        sns.set()
        # plot the entire dataset in blue
        warnings.filterwarnings("ignore")
        plt.figure(0,figsize=[12,6])
        plt.plot(series.values, label='True time-series')

        # if self._n_seq == 1:
        # plot the forecasts
        for i in range(len(forecasts)):
            off_s = len(series) - n_test + i
            off_e = off_s + len(forecasts[i])
            xaxis = [x for x in range(off_s, off_e)]
            if i==0:
                lb = 'Forecasted time-series'
            else:
                lb = None
            sns.lineplot(x=xaxis, y=forecasts[i], label=lb,color='r',hue_order=False)
            #plt.plot(xaxis, forecasts[i], color='red',label='Forecasted time-series')
        # show the plot
        plt.show()
        sns.reset_defaults()