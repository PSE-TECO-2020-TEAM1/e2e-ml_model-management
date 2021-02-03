from abc import ABC, abstractmethod
from typing import List
from pandas import DataFrame
from numpy import ndarray

class IImputer(ABC):

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> List[DataFrame]:
        pass


class INormalizer(ABC):

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        pass


class IClassifier(ABC):

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, df: DataFrame) -> ndarray:
        pass
