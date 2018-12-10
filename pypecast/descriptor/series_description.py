from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss

import warnings

class SeriesDescriptor(object):
    '''Class for quickly analyzing a time series. '''
    def __init__(self):
        pass
    
    def full_report(self, data, ppy=12):
        self.describe(data)
        self.autocorr(data)
        self.trend_check(data)
        self.seasonality_test(data, ppy)
        self.outliers_check(data)
        self.lr_cycle_check(data)
        self.const_var_check(data)
        self.get_abrupt_changes(data)

    def describe(self, data):
        print('-> Description of the series data:')
        if isinstance(data,(pd.DataFrame,)):
            df = data
            print(df.describe())
        else:
            df = pd.DataFrame(data)
            print(df.describe()) 

        print('-'*20 + ' Histogram ' + '-'*20)
        df.hist()

    def autocorr(self, data, unbiased = False, ):
        #configuration of the plot
        print('-> Autocorrelation and partial autocorrelation:')
        warnings.filterwarnings("ignore")
        sns.set()
        plt.figure(figsize=[12,6])

        #Autocorrelation array
        ac = acf(data)
        
        plot_acf(data)

        plt.title('Autocorrelation of the series')
        plt.xlabel('timestep')
        plt.ylabel('autocorrelation')

        plot_pacf(data)

        plt.title('Partial autocorrelation of the series')
        plt.xlabel('timestep')
        plt.ylabel('Partial autocorrelation')
        sns.reset_defaults()
        return ac
        
    def stationarity_kpps(self,data):
        return kpps(data)
    
    def trend_check(self, data):
        print('-> Checking for trends:')
        
    
    def seasonality_test(self, data, ppy=12):
        """
        Seasonality test
        :param data: time series
        :param ppy: periods per year
        :return: boolean value: whether the TS is seasonal
        """
        s = acf(data, 1)
        for i in range(2, ppy):
            s = s + (acf(data, i) ** 2)

        limit = 1.645 * (np.sqrt((1 + 2 * s) / len(data)))

        return (abs(acf(data, ppy))) > limit

    
    def outliers_check(self, data):
        print('-> Checking for outliers:')
        

    def lr_cycle_check(self, data):
        print('-> Checking for long-run cycles:')
    
    def const_var_check(self, data):
        print('-> Checking if series variance is constant:')

    def get_abrupt_changes(self, data):
        print('-> Checking for abrupt changes:')
        