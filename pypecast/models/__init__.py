from pypecast.models.Dense import fit_dense
from pypecast.models.LSTM import fit_lstm

models = dict(
    fit_dense=fit_dense,
    fit_lstm=fit_lstm,
)

__all__ = [
    'fit_dense',
    'fit_lstm',
    'models'
]