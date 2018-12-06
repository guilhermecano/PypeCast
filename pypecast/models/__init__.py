from pypecast.models.mlp import MLP
from pypecast.models.simple_lstm import Simple_LSTM
from pypecast.models.model import Model

models = dict(
    MLP=MLP,
    Simple_LSTM=Simple_LSTM
)

__all__ = [
    'Model',
    'Simple_LSTM',
    'MLP',
    'models'
]