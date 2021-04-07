from app.ml.objects.imputation import Imputation
from sklearn.impute import SimpleImputer
from pyts.preprocessing import InterpolationImputer

imputer_factory_dict = {
    Imputation.MEAN_IMPUTATION: lambda: SimpleImputer(strategy="mean"),
    Imputation.ZERO_INTERPOLATION: lambda: InterpolationImputer(strategy="zero"),
    Imputation.LINEAR_INTERPOLATION: lambda: InterpolationImputer(strategy="linear"),
    Imputation.QUADRATIC_INTERPOLATION: lambda: InterpolationImputer(strategy="quadratic"),
    Imputation.CUBIC_INTERPOLATION: lambda: InterpolationImputer(strategy="cubic")
    # TODO add these imputations
    # Imputation.MOVING_AVERAGE_IMPUTATION: ,
    # Imputation.LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION:
}

def get_imputer(imputation: Imputation):
    return imputer_factory_dict[imputation]()