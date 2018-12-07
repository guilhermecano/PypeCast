from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

class SeriesDescriptor(object):
    '''Class for quickly analyzing a time series. '''
    def __init__(self):
        pass
    
    def full_report(self, data):
        self.describe(data)
        self.autocorr(data)
        self.trend_check(data)
        self.seasonality_check(data)
        self.outliers_check(data)
        self.lr_cycle_check(data)
        self.const_var_check(data)
        self.get_abrupt_changes(data)

    def describe(self, data):
        print('-> Description of the series data:')
        if isinstance(data,(pd.DataFrame,)):
            print(data.describe())
        else:
            print(pd.DataFrame(data).describe()) 

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
        
    def trend_check(self, data):
        print('-> Checking for trends:')
        
    
    def seasonality_check(self, data):
        print('-> Checking for seasonality:')
    
    def outliers_check(self, data):
        print('-> Checking for outliers:')

    def lr_cycle_check(self, data):
        print('-> Checking for long-run cycles:')
    
    def const_var_check(self, data):
        print('-> Checking if series variance is constant:')

    def get_abrupt_changes(self, data):
        print('-> Checking for abrupt changes:')
        