import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from pandas import DataFrame

_TARGET_DIR = Path(__file__).resolve().parent.parent.parent / 'target'


def load_preprocessor() -> any:
    with open(_TARGET_DIR / 'preprocessor.pkl', 'rb') as file:
        return pickle.load(file)


class DataPreprocessor(ABC):

    @abstractmethod
    def preprocess_data(self, X: DataFrame):
        """
        Preprocesses the data.
        :return:
        """

    def save_preprocessor(self):
        _TARGET_DIR.mkdir(parents=True, exist_ok=True)

        with open(_TARGET_DIR / 'preprocessor.pkl', 'wb') as file:
            pickle.dump(self, file)
