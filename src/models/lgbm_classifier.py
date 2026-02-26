import pandas as pd
from pandas import DataFrame
from hyperopt import hp
from lightgbm import LGBMClassifier, early_stopping

from src.enums.objective import Objective
from src.models.model_wrapper import ModelWrapper


class LGBMClassifierWrapper(ModelWrapper):

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
        return LGBMClassifier(
            verbose=-1,
            **params
        )

    def get_starter_params(self) -> dict:
        return {
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'num_leaves': 64,
            'max_depth': 10,  # was -1 (better?)
            'learning_rate': 0.1,
            'n_estimators': 20000,
            'min_child_samples': 20,
            'reg_alpha': 0.2,
            'reg_lambda': 5,
            'colsample_bytree': 0.6,  # was 1.0, sometimes better
            "colsample_bynode": 0.6,  # new
            "extra_trees": True,  # new (better without, but helps reduce overfitting)
            "max_bin": 255,  # new
            'subsample': 1.0,
            'n_jobs': -1,
            'random_state': 0
        }

    def get_grid_space(self) -> list[dict]:
        return [
            {
                'recalibrate_iterations': False,
                'max_depth': list(range(3, 10)),
                # 'min_child_weight': range(1, 6)
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

    def get_bayesian_space(self) -> dict:
        return {
            # 'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            # 'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            "num_leaves": hp.quniform("num_leaves", 8, 64, 1),  # was 10-150
            "max_depth": hp.quniform("max_depth", 3, 12, 1),
            "min_child_samples": hp.quniform("min_child_samples", 5, 100, 1),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            # "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0), # yun confirmation
            # "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0), # yun confirmation
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1)
        }

    def fit(self, X, y, iterations, params=None):
        params = params or {}
        params = params.copy()
        params.update({
            'random_state': 0,
            'n_estimators': iterations,
        })

        self.model: LGBMClassifier = LGBMClassifier(
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
        self.model: LGBMClassifier = LGBMClassifier(
            **params
        )
        self.model.fit(train_X, train_y, eval_set=[(validation_X, validation_y)], callbacks=[
            early_stopping(stopping_rounds=self.early_stopping_rounds),
        ])

    def predict(self, X) -> any:
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

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
