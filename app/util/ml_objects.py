from abc import ABC, abstractmethod
from typing import List

from numpy import ndarray
from pandas import DataFrame


class IImputer(ABC):

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> ndarray:
        pass


class INormalizer(ABC):

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> ndarray:
        pass

class IClassifier(ABC):

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, df: DataFrame, y: DataFrame) -> ndarray:
        pass
