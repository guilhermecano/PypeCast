from keras.models import Sequential,Graph
from keras.layers.core import Dense, Dropout
from keras.callbacks import History
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras.regularizers import l2, activity_l2
from keras import backend as K

from pypecast.models import Model
from pypecast.metrics import MDNCollection

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

        collection = MDNCollection(self._n_seq, self._n_distr)
        self._loss = collection.mean_log_Gaussian_like
    
   def _design_network(self, inp_shape, out_shape):
       '''Design an architecture for time-series forecasting with uncertainty modelling'''
        graphG = Graph()
        graphG.add_input(name='input', input_shape=inp_shape, dtype='float32')
        graphG.add_node(LSTM(output_dim=5, return_sequences=True), name='LSTM1_1', input='input')
        graphG.add_node(Dropout(0.5), name='Dropout1', input='LSTM1_1')
        graphG.add_node(Dense(output_dim=5, activation="relu"), name='FC1', input='Dropout1')
        graphG.add_node(Dense(output_dim=self._n_seq*self._n_distr), name='FC_mus', input='Dropout1')
        graphG.add_node(Dense(output_dim=self._n_distr, activation=K.exp, W_regularizer=l2(1e-3)), name='FC_sigmas', input='FC1')
        graphG.add_node(Dense(output_dim=self._n_distr, activation='softmax'), name='FC_alphas', input='FC1')
        graphG.add_output(name='output', inputs=['FC_mus','FC_sigmas', 'FC_alphas'], merge_mode='concat',concat_axis=1)
        graphG.compile(optimizer='rmsprop', loss={'output':self._loss)

    def fit(self, train,n_batch = 1, n_epoch = 1000, early_stopping = False):
        '''Fit model to training data'''
        
        
        
