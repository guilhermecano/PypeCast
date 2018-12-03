from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, h1,
                 h2=None, dropout=0.5, actv = 'relu',
                 loss_func = 'mse', opt = 'adam', es = None):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1]) #n_linhas,  n_timesteps, n_features
    # design network
    model = Sequential()
    model.add(LSTM(h1))
    if h2 is not None:
        model.add(Dense(h2,activation=actv))
    model.add(Dense(y.shape[1]))
    model.compile(loss=loss_func, optimizer=opt)
    # fit network
    model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=2, 
                shuffle=False, validation_split=0.2, callbacks=es)
    return model