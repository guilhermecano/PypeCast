from pypecast.metrics.mdn_collection import MDNCollection
from pypecast.metrics.metrics import rmse,mae,mse,mape,smape,maek

metrics = dict(
    MDNCollection=MDNCollection,
    rmse = rmse,
    mae = mae,
    mse = mse,
    mape = mape,
    smape = smape,
    maek = maek
)

__all__ = [
    'metrics'
]