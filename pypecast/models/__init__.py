from pypecast.models.Dense import fit_dense
from pypecast.models.simple_lstm import Simple_LSTM
from pypecast.models.model import Model

models = dict(
    Dense=fit_dense,
    LSTM=Simple_LSTM
)

__all__ = [
    'Model',
    'Simple_LSTM',
    'fit_dense',
    'models'
]