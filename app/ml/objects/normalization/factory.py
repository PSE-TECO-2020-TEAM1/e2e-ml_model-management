from app.ml.objects.normalization import Normalization
from sklearn.preprocessing import (MinMaxScaler,
                                   Normalizer,
                                   QuantileTransformer,
                                   RobustScaler,
                                   StandardScaler)


normalizer_factory_dict = {
    Normalization.MIN_MAX_SCALER: lambda: MinMaxScaler(),
    Normalization.NORMALIZER: lambda: Normalizer(),
    Normalization.QUANTILE_TRANSFORMER: lambda: QuantileTransformer(),
    Normalization.ROBUST_SCALER: lambda: RobustScaler(),
    Normalization.STANDARD_SCALER: lambda: StandardScaler()
}


def get_normalizer(normalization: Normalization):
    return normalizer_factory_dict[normalization]()
