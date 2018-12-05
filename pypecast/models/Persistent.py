# make a persistent forecast
def persistent(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]