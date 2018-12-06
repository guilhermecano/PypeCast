from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd

class SeriesDescriptor(object):
    def __init__(self):
        pass
    
    def full_report(self, data):
        raise(NotImplementedError)

    def describe(self, data):
        print('Description of the series data:')
        if isinstance(data,(pd.DataFrame,)):
            print(data.describe())
        else:
            print(pd.DataFrame(data).describe()) 

    def autocorrelation(self, data):
        raise(NotImplementedError)

    def trend_check(self, data):
        raise(NotImplementedError)
    
    def seasonality_check(self, data):
        raise(NotImplementedError)
    
    def outliers_check(self, data):
        raise(NotImplementedError)

    def lr_cycle_check(self, data):
        raise(NotImplementedError)
    
    def const_var_check(self, data):
        raise(NotImplementedError)

    def get_abrupt_changes(self, data):
        raise(NotImplementedError)