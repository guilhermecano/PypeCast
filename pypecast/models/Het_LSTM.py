from keras.models import Sequential,Graph
from keras.layers.core import Dense, Dropout
from keras.callbacks import History
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras.regularizers import l2, activity_l2
from keras import backend as K

from pypecast.models import Model

class MDN_Het_LSTM(Model):
    '''Class for defining and training a common LSTM network for time series forecasting'''
    def __init__(self, 
                n_lag, 
                n_seq):

        super(MDN_Het_LSTM, self).__init__(
            n_lag = n_lag,
            n_seq=n_seq
        )
    
   def _design_network(self, inp_shape, out_shape):
       '''Design an architecture for time-series forecasting with uncertainty modelling'''
        ograph = Graph()
        ograph.add_input(name='input', input_shape=inp_shape, dtype='float32')
        ograph.add_node(LSTM(output_dim=128, return_sequences=True), name='LSTM1_1', input='input')
        # ograph.add_node(Dropout(0.5), name='Dropout1', input='LSTM1_1')
        # ograph.add_node(LSTM(output_dim=128, return_sequences=False), name='LSTM2_1', input='Dropout1')
        # ograph.add_node(Dropout(0.5), name='Dropout2', input='LSTM2_1')
        # ograph.add_node(Dense(output_dim=128, activation="relu"), name='FC1', input='Dropout2')
        # ograph.add_node(Dense(output_dim=out_shape, activation="linear"), name='FC2', input='FC1')
        ograph.add_output(name='output', input='FC2')
        ograph.compile(optimizer='rmsprop', loss={'output':'mean_absolute_error'})
