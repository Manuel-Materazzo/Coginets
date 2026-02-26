import os
import pickle
import pandas as pd
import json
from abc import ABC, abstractmethod
from pathlib import Path
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config

from src.utils.json_utils import map_dtype

_TARGET_DIR = Path(__file__).resolve().parent.parent.parent / 'target'


def load_pipeline() -> any:
    with open(_TARGET_DIR / 'pipeline.pkl', 'rb') as file:
        return pickle.load(file)


def save_data_model(X: DataFrame):
    config = {
        "fields": {col: map_dtype(dtype) for col, dtype in X.dtypes.items()}
    }

    _TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Save config dictionary as JSON file
    with open(_TARGET_DIR / 'data-model.json', 'w+') as f:
        json.dump(config, f, indent=4)


class DTPipeline(ABC):

    def __init__(self, X: DataFrame):
        set_config(transform_output="pandas")
        # Select categorical columns
        self.categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
        # Select numerical columns
        self.numerical_cols = [cname for cname in X.columns if pd.api.types.is_numeric_dtype(X[cname])]

        self.pipeline = self.build_pipeline()

    @abstractmethod
    def build_pipeline(self) -> Pipeline | ColumnTransformer:
        """
        Builds the pipeline
        :return:
        """

    def fit_transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Wrapping function to fit and transform the dataframe.
        :param dataframe:
        :return:
        """
        set_config(transform_output="pandas")
        imputed_dataframe = pd.DataFrame(self.pipeline.fit_transform(dataframe))
        return imputed_dataframe

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Wrapping function to transform the dataframe.
        :param dataframe:
        :return:
        """
        set_config(transform_output="pandas")
        imputed_dataframe = pd.DataFrame(self.pipeline.transform(dataframe))
        return imputed_dataframe

    def get_pipeline_with_training(self, model: any) -> Pipeline:
        """
        Gets the pipeline with the added model training step
        :param model:
        :return:
        """
        return Pipeline(steps=[
            ('preprocessor', self.pipeline),
            ('model', model)
        ])

    def find_one_hot_encoder(self, transformer):
        """
        Checks if the pipeline contains one hot encoder and returns the instance.
        :param transformer:
        :return:
        """
        if isinstance(transformer, Pipeline):
            for name, step in transformer.steps:
                result = self.find_one_hot_encoder(step)
                if result is not None:
                    return result
        elif isinstance(transformer, ColumnTransformer):
            for name, trans, columns in transformer.transformers:
                result = self.find_one_hot_encoder(trans)
                if result is not None:
                    return result
        elif isinstance(transformer, OneHotEncoder):
            return transformer
        return None

    def save_pipeline(self):
        _TARGET_DIR.mkdir(parents=True, exist_ok=True)

        with open(_TARGET_DIR / 'pipeline.pkl', 'wb') as file:
            pickle.dump(self, file)
