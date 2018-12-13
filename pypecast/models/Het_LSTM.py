from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.models import Model as Md
from keras.layers import Dense, Dropout, Input, LSTM, concatenate
from keras.callbacks import History
# from keras.models import model_from_json
from keras.regularizers import l2
from keras import backend as K
import numpy as np
from pypecast.models import Model
from pypecast.metrics import MDNCollection
from pypecast.metrics.metrics import *

from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

class MDN_Het_LSTM(Model):
    '''Class for defining and training a common LSTM network for time series forecasting'''
    def __init__(self, 
                n_lag, 
                n_seq,
                n_distr=1):

        super(MDN_Het_LSTM, self).__init__(
            n_lag = n_lag,
            n_seq=n_seq
        )

        self._n_distr = n_distr

        self._stds = None

        collection = MDNCollection(self._n_seq, self._n_distr)
        self._loss = collection.mean_log_Gaussian_like

    def _design_network(self, inp_shape):
        '''Design an architecture for time-series forecasting with uncertainty modelling'''
        inpt = Input(inp_shape)
        modelG = LSTM(units=5, name='LSTM_1')(inpt)
        #modelG = Dense(units=5, activation='relu',name='Dense1')(modelG)
        
        mu_units = Dense(units=self._n_seq*self._n_distr)(modelG)
        std_unit = Dense(units=self._n_distr, activation=K.exp)(modelG)
        alpha_unit = Dense(units=self._n_distr, activation='softmax')(modelG)

        out = concatenate([mu_units, std_unit, alpha_unit])

        model = Md(inputs=inpt, outputs=out)

        self._model = model

    def fit(self, train,n_batch = 1, n_epoch = 1000, early_stopping = False):
        '''Fit model to training data'''
        X, y = train[:, 0:self._n_lag], train[:, self._n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1]) #n_series,  n_timesteps, n_features

        # design network
        self._design_network(inp_shape=X.shape[1:])
        # #compile model
        self._model.compile(loss=self._loss, optimizer='adam')
        self._model.summary()
        #check early stopping
        if isinstance(early_stopping, (bool,)):
            #default es
            if early_stopping:
                es = [EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None)]
            else:
                es = early_stopping
        else:
            es = [early_stopping]
        # fit network
        self._model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2, shuffle=False, validation_split=0.2, callbacks=es)
        
        print('-'*60)
        print('Model trained')

    def _forecast_model(self, X):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = self._model.predict(X)
        # convert to array
        f = [x for x in forecast[0, :self._n_seq]]
        s = forecast[0, self._n_seq]
        return f,s
    
    def forecast_series(self, test, scaler, orig_series):
        assert self._model is not None, "Model must be trained first" 
        if isinstance(orig_series,(list,)):
            orig_series = np.array(orig_series)
        if isinstance(test,(list,)):
            test = np.array(test)
        
        forecasts = list()
        stds = list()
        for i in range(len(test)):
            X, y = test[i, 0:self._n_lag], test[i, self._n_lag:]
            # make forecast
            forecast, sd = self._forecast_model(X)
            
            # store the forecast
            forecasts.append(forecast)
            stds.append(sd)
        
        self._naive_init = orig_series[orig_series.shape[0] - test.shape[0] - self._n_seq]

        #inverse_transform
        forecasts = self._inverse_transform(orig_series,forecasts,scaler,test.shape[0]+2)
        self._stds = scaler[0].inverse_transform(stds)
        print(stds)
        self._forecasts = forecasts
        #Actual values
        actual = [row[self._n_lag:] for row in test]
        self._actual = self._inverse_transform(orig_series, actual, scaler, test.shape[0]+2)

        return forecasts
    def _use_metrics_wkeep(self, actual, predicted, kr=0.5):
            k = kr*np.max(self._stds)
            #RMSE
            m1 = rmse(actual, predicted)
            #MAE
            m2 = mae(actual, predicted)
            #MAPE
            m3 = mape(actual, predicted)
            #sMAPE
            m4 = smape(actual, predicted)
            #MAE_keep
            m5 = maek(actual, predicted, k)
            return m1,m2,m3,m4,m5

    def evaluate_forecast(self, save_report = False, filename = '../reports/evaluation.xlsx', return_dicts = False, verbose = 1):
        if verbose!=0:
            print('-'*20 + 'Forecast evaluation' + '-'*20)
            print('')
        steps_metrics = dict()
        naive_metrics = dict()
        instant_metrics = {'RMSE':[], 'MAE':[], 'MAPE':[], 'sMAPE': [], 'MAEk': []}
        
        # Metrics for each timestep in future
        for i in range(self._n_seq):
            if verbose!=0:
                print('Step t+{}'.format(i+1))
            actual = [row[i] for row in self._actual]        
            #print(np.array(actual))
            predicted = [forecast[i] for forecast in self._forecasts]

            m1,m2,m3,m4 = self._use_metrics(actual,predicted)
            if verbose!=0:
                print('t+%d RMSE: %f' % ((i+1), m1))
                print('t+%d MAE: %f' % ((i+1), m2))
                print('t+%d MAPE: %f' % ((i+1), m3))
                print('t+%d sMAPE: %f' % ((i+1),m4))

            steps_metrics[(i+1)] = [m1,m2,m3,m4]
            if verbose!=0:
                print('-'*60)
            
        # Metrics for naive_model:
        if verbose!=0:
            print()
            print('-'*20 + 'Naive forecast evaluation' + '-'*20)
        #Get persistent-series frocasts
        last_ob = [row[0] for row in self._actual]
        last_ob.pop()
        last_ob.insert(0,self._naive_init)
        #Evaluate the persistent case
        naive_forecasts = list()
        for i in last_ob:
            lst = [i]*self._n_seq
            naive_forecasts.append(lst)
        
        for i in range(self._n_seq):
            if verbose!=0:
                print('Step t+{}'.format(i+1))
            actual = [row[i] for row in self._actual]
            naive = [nf[i] for nf in naive_forecasts]

            m1,m2,m3,m4 = self._use_metrics(actual,naive)
            if verbose!=0:
                print('t+%d RMSE: %f' % ((i+1), m1))
                print('t+%d MAE: %f' % ((i+1), m2))
                print('t+%d MAPE: %f' % ((i+1), m3))
                print('t+%d sMAPE: %f' % ((i+1),m4))

            naive_metrics[(i+1)] = [m1,m2,m3,m4]
            if verbose!=0:
                print('-'*60)

        if verbose!=0:
            print()
            print('-'*20 + 'Evaluation for each forecast' + '-'*20)
        # Metrics for each instant in time-series 
        for i in range(len(self._actual)):

            m1,m2,m3,m4,m5 = self._use_metrics_wkeep(self._actual[i],self._forecasts[i])
            if verbose!=0:    
                print('Index %d RMSE: %f' % ((i+1), m1))
                print('Index %d MAE: %f' % ((i+1), m2))
                print('Index %d MAPE: %f' % ((i+1), m3))
                print('Index %d sMAPE: %f' % ((i+1),m4))
                print('Index %d MAE Keep: %f' % ((i+1),m5))

            instant_metrics['RMSE'].append(m1)
            instant_metrics['MAE'].append(m2)
            instant_metrics['MAPE'].append(m3)
            instant_metrics['sMAPE'].append(m4)
            instant_metrics['MAEk'].append(m5)

            if verbose != 0:
                print('-'*60)

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
            if self._n_seq>1:
                #sns.lineplot(x=xaxis, y=forecasts[i], label=lb,color='r',hue_order=False)
                plt.errorbar(x=xaxis, y=forecasts[i], yerr=self._stds[i], linestyle='None', marker='^', color='r')
            else:
                #sns.scatterplot(x=xaxis, y=forecasts[i], label=lb,color='r',hue_order=False)
                plt.errorbar(x=xaxis, y=forecasts[i], yerr=self._stds[i], linestyle='None', marker='^', color='r')
        # show the plot
        plt.show()
        sns.reset_defaults()