import pandas as pd
from pandas import DataFrame
from hyperopt import hp
from xgboost import XGBClassifier

from src.enums.objective import Objective
from src.models.model_wrapper import ModelWrapper


class XGBClassifierWrapper(ModelWrapper):

    def __init__(self, early_stopping_rounds=10):
        super().__init__(early_stopping_rounds=early_stopping_rounds)

    def get_objective(self) -> Objective:
        return Objective.CLASSIFICATION

    def get_base_model(self, iterations, params):
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': iterations,
        })
        return XGBClassifier(
            **params
        )

    def get_starter_params(self) -> dict:
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1
        }

    def get_grid_space(self) -> list[dict]:
        return [
            {
                'recalibrate_iterations': False,
                'max_depth': list(range(3, 10)),
                'min_child_weight': list(range(1, 6))
            },
            {
                'recalibrate_iterations': False,
                'gamma': [i / 10.0 for i in range(0, 5)]
            },
            {
                'recalibrate_iterations': True,
                'subsample': [i / 100.0 for i in range(60, 100, 5)],
                'colsample_bytree': [i / 100.0 for i in range(60, 100, 5)]
            },
            {
                'recalibrate_iterations': False,
                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
            }
        ]

    def get_bayesian_space(self) -> dict:
        return {
            'max_depth': hp.quniform("max_depth", 3, 10, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'gamma': hp.uniform('gamma', 0, 0.5),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        }

    def fit(self, X, y, iterations, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': iterations,
        })

        self.model: XGBClassifier = XGBClassifier(
            **params
        )

        self.model.fit(X, y)

    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': 2000,
            'early_stopping_rounds': self.early_stopping_rounds,
        })
        self.model: XGBClassifier = XGBClassifier(
            **params
        )
        self.model.fit(train_X, train_y, eval_set=[(validation_X, validation_y)], verbose=False)

    def predict(self, X) -> any:
        return self.model.predict(X)

    def predict_proba(self, X) -> any:
        return self.model.predict_proba(X)[:, 1]

    def get_best_iteration(self) -> int:
        return self.model.best_iteration

    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        if self.model is None:
            raise ValueError("No model has been fitted")

        return self.model.evals_result()

    def get_feature_importance(self, features) -> DataFrame:
        if self.model is None:
            raise ValueError("No model has been fitted")

        importances = self.model.feature_importances_

        # sort and merge importances and column names into a dataframe
        feature_importances = sorted(zip(importances, features), reverse=True)
        sorted_importances, sorted_features = zip(*feature_importances)
        return pd.DataFrame({'feats': sorted_features[:50], 'importance': sorted_importances[:50]})
