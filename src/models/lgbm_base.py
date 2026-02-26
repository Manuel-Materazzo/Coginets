import pandas as pd
from abc import abstractmethod
from pandas import DataFrame
from hyperopt import hp
from lightgbm import early_stopping

from src.models.model_wrapper import ModelWrapper


class LGBMBaseWrapper(ModelWrapper):

    def __init__(self, early_stopping_rounds=10):
        super().__init__(early_stopping_rounds=early_stopping_rounds)

    @abstractmethod
    def _get_model_class(self):
        """Returns the LightGBM model class (LGBMRegressor or LGBMClassifier)."""

    def get_base_model(self, iterations, params):
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': iterations,
        })
        return self._get_model_class()(
            verbose=-1,
            **params
        )

    def get_grid_space(self) -> list[dict]:
        return [
            {
                'recalibrate_iterations': False,
                'max_depth': list(range(3, 10)),
            },
            {
                'recalibrate_iterations': True,
                'subsample': [i / 100.0 for i in range(60, 100, 5)],
                'colsample_bytree': [i / 100.0 for i in range(60, 100, 5)]
            },
            {
                'recalibrate_iterations': False,
                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            }
        ]

    def fit(self, X, y, iterations, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': iterations,
        })

        self.model = self._get_model_class()(
            **params
        )

        self.model.fit(X, y)

    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': 2000,
        })
        self.model = self._get_model_class()(
            **params
        )
        self.model.fit(train_X, train_y, eval_set=[(validation_X, validation_y)], callbacks=[
            early_stopping(stopping_rounds=self.early_stopping_rounds),
        ])

    def predict(self, X) -> any:
        return self.model.predict(X)

    def get_best_iteration(self) -> int:
        return self.model.best_iteration_

    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        if self.model is None:
            raise ValueError("No model has been fitted")

        return self.model.evals_result_

    def get_feature_importance(self, features) -> DataFrame:
        if self.model is None:
            raise ValueError("No model has been fitted")

        importances = self.model.feature_importances_

        # sort and merge importances and column names into a dataframe
        feature_importances = sorted(zip(importances, features), reverse=True)
        sorted_importances, sorted_features = zip(*feature_importances)
        return pd.DataFrame({'feats': sorted_features[:50], 'importance': sorted_importances[:50]})
