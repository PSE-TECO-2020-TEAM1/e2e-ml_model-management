from sklearn.compose import ColumnTransformer
from pandas import DataFrame

class PandasColumnTransformer(ColumnTransformer):
    def transform(self, X: DataFrame) -> DataFrame:
        return DataFrame(super().transform(X), columns=X.columns, index=X.index)

    def fit_transform(self, X: DataFrame, y=None) -> DataFrame:
        return DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)