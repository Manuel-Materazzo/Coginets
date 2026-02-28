from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipelines.dt_pipeline import DTPipeline
from src.utils.logger import log


class FunctionalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        try:
            X['Functional'] = X['Functional'].fillna('Typ')  # Assume typical unless deductions are warranted
        except KeyError:
            log.warning("KeyError: 'Functional' column not found")

        return X


class HousingPricesCompetitionDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame):
        super().__init__(X)

    def build_pipeline(self) -> Pipeline | ColumnTransformer:

        # Encoding for categorical data
        categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', categorical_encoder)
        ], memory=None)

        preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
        ])

        # Bundle preprocessing
        return Pipeline(steps=[
            ('functional_imputer', FunctionalImputer()),
            ('preprocessor', preprocessor)
        ], memory=None)