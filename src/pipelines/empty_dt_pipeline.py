from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.pipelines.dt_pipeline import DTPipeline


class Transform(TransformerMixin):
    def __init__(self, **kwargs):
        self.hyperparam = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class EmptyDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame):
        super().__init__(X)

    def build_pipeline(self) -> Pipeline | ColumnTransformer:
        return Pipeline(steps=[('do nothing', Transform())], memory=None)
