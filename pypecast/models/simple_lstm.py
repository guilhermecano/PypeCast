from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

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
        #Must have variables
        self.n_lag = n_lag
        self.n_seq = n_seq
        
        #Optional architecture
        self.n_units = n_units
        self.hidden = hidden
        self.dropout = dropout
        self.activation = hidden_activation

    def fit(self, train, n_batch, nb_epoch, loss = 'mse', opt = 'adam', early_stopping = False):
        X, y = train[:, 0:self.n_lag], train[:, self.n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1]) #n_series,  n_timesteps, n_features

        # design network
        model = Sequential()
        model.add(LSTM(self.n_units))
        if self.hidden is not None:
            #verify if it is list
            assert isinstance(self.hidden, (list,))
            #check if dropout was defined individually
            if len(self.hidden) == len(self.dropout):
                for n, drop in zip(self.hidden,self.dropout):
                    model.add(Dense(n,activation=self.activation))
                    model.add(Dropout(drop))
            elif self.dropout is None:
                for n in self.hidden:
                    model.add(Dense(n,activation=self.activation))        
        model.add(Dense(y.shape[1]))
        model.compile(loss=loss, optimizer=opt)

        #check early stopping
        if isinstance(early_stopping, (bool,)):
            #default es
            if early_stopping:
                es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None)
        else:
            es = [early_stopping]
        # fit network
        model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=2, 
                    shuffle=False, validation_split=0.2, callbacks=es)
        self._model = model
        
        print('-'*60)
        print('Model trained')
        
    def _forecast_model(self, X):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = self._model.predict(X)
        # convert to array
        return [x for x in forecast[0, :]]

    def forecast_series(self, test):
        assert self._model is not None, "Model must be trained first"

        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:self.n_lag], test[i, self.n_lag:]
            # make forecast
            forecast = self._forecast_model(X)
            # store the forecast
            forecasts.append(forecast)
        self._forecasts = forecasts

        