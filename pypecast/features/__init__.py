from pypecast.features.build_features import BuildFeatures
from pypecast.features.features_supervised import BuildFeaturesSupervised

features = dict(
    BuildFeatures=BuildFeatures,
    BuildFeaturesSupervised = BuildFeaturesSupervised
)

__all__ = [
    'features'
]