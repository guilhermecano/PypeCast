from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import keras.backend as K
import numpy as np

class MDNCollection(object):
    def __init__(self, n_seq, n_distr = 1):
        self._n_seq = n_seq  #The number of outputs we want to predict
        self._n_distr = n_distr  #The number of distributions we want to use in the mixture

    def log_sum_exp(self, x, axis=None):
        """Log-sum-exp trick implementation"""
        x_max = K.max(x, axis=axis, keepdims=True)
        return K.log(K.sum(K.exp(x - x_max), 
                        axis=axis, keepdims=True))+x_max

    def mean_log_Gaussian_like(self, y_true, parameters):
        """Mean Log Gaussian Likelihood distribution
        Note: The 'self._n_seq' variable is obtained as a private class variable
        """
        components = K.reshape(parameters,[-1, self._n_seq + 2, self._n_distr])
        mu = components[:, :self._n_seq, :]
        sigma = components[:, self._n_seq, :]
        alpha = components[:, self._n_seq + 1, :]
        alpha = K.softmax(K.clip(alpha,1e-8,1.))
        
        exponent = K.log(alpha) - .5 * float(self._n_seq) * K.log(2 * np.pi) \
        - float(self._n_seq) * K.log(sigma) \
        - K.sum((K.expand_dims(y_true,2) - mu)**2, axis=1)/(2*(sigma)**2)
        
        log_gauss = self.log_sum_exp(exponent, axis=1)
        res = - K.mean(log_gauss)
        return res

    def mean_log_LaPlace_like(self, y_true, parameters):
        """Mean Log Laplaself._n_seqe Likelihood distribution
        Note: The 'self._n_seq' variable is obtained as a private class variable
        """
        components = K.reshape(parameters,[-1, self._n_seq + 2, self._n_distr])
        mu = components[:, :self._n_seq, :]
        sigma = components[:, self._n_seq, :]
        alpha = components[:, self._n_seq + 1, :]
        alpha = K.softmax(K.clip(alpha,1e-2,1.))
        
        exponent = K.log(alpha) - float(self._n_seq) * K.log(2 * sigma) \
        - K.sum(K.abs(K.expand_dims(y_true,2) - mu), axis=1)/(sigma)
        
        log_gauss = self.log_sum_exp(exponent, axis=1)
        res = - K.mean(log_gauss)
        return res