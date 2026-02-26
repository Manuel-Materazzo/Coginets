import pandas as pd
from pandas import DataFrame
from hyperopt import hp
from catboost import CatBoostClassifier

from src.enums.objective import Objective
from src.models.model_wrapper import ModelWrapper


class CatBoostClassifierWrapper(ModelWrapper):

    def __init__(self, early_stopping_rounds=10):
        super().__init__(early_stopping_rounds=early_stopping_rounds)

    def get_objective(self) -> Objective:
        return Objective.CLASSIFICATION

    def get_base_model(self, iterations, params):
        params = params.copy()
        params.update({
            'random_state': 0,
            'iterations': iterations,
        })
        return CatBoostClassifier(
            **params,
            silent=True
        )

    def get_starter_params(self) -> dict:
        return {
            'loss_function': 'Logloss',
            # 'bootstrap_type': 'Bayesian',  # removed, let catboost decide
            'grow_policy': 'SymmetricTree',
            'bagging_temperature': 0.50,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 1.25,  # was 3.0
            'min_data_in_leaf': 24,
            # 'random_strength': 0.25, # ignored, prevents overfitting, but it's not mandatory
            'thread_count': -1
        }

    def get_grid_space(self) -> list[dict]:
        return [
            {
                'recalibrate_iterations': False,
                'depth': [4, 6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            },
            {
                'recalibrate_iterations': True,
                'bagging_temperature': [0.1, 0.5, 1, 2],
                # 'subsample': [0.6, 0.8, 1.0] # bayesian bootstrap doesn't support 'subsample' option
            },
            {
                'recalibrate_iterations': False,
                'random_strength': [0, 0.5, 1, 2, 5]
            }
        ]

    def get_bayesian_space(self) -> dict:
        return {
            'depth': hp.quniform('depth', 4, 10, 1),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 10),
            # 'l2_leaf_reg': Real(1e-2, 10, prior='log-uniform'),
            'bagging_temperature': hp.uniform('bagging_temperature', 0.1, 2.0),
            # 'subsample': hp.uniform('subsample', 0.5, 1.0, ), # bayesian bootstrap doesn't support 'subsample' option
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1.0),
            'random_strength': hp.uniform('random_strength', 0, 10),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 100, 1),
            'max_bin': hp.quniform('max_bin', 100, 500, 1)
        }

    def fit(self, X, y, iterations, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'iterations': iterations,
        })

        self.model: CatBoostClassifier = CatBoostClassifier(
            **params,
            silent=True
        )

        self.model.fit(X, y, silent=True)

    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'iterations': 2000,
            'early_stopping_rounds': self.early_stopping_rounds,
        })
        self.model: CatBoostClassifier = CatBoostClassifier(
            **params,
            silent=True
        )
        self.model.fit(train_X, train_y, eval_set=[(validation_X, validation_y)], silent=True)

    def predict(self, X) -> any:
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_best_iteration(self) -> int:
        return self.model.get_best_iteration()

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
