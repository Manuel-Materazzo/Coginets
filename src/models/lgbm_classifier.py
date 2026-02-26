from hyperopt import hp
from lightgbm import LGBMClassifier

from src.enums.objective import Objective
from src.models.lgbm_base import LGBMBaseWrapper


class LGBMClassifierWrapper(LGBMBaseWrapper):

    def _get_model_class(self):
        return LGBMClassifier

    def get_objective(self) -> Objective:
        return Objective.CLASSIFICATION

    def get_starter_params(self) -> dict:
        return {
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'num_leaves': 64,
            'max_depth': 10,
            'learning_rate': 0.1,
            'n_estimators': 20000,
            'min_child_samples': 20,
            'reg_alpha': 0.2,
            'reg_lambda': 5,
            'colsample_bytree': 0.6,
            "colsample_bynode": 0.6,
            "extra_trees": True,
            "max_bin": 255,
            'subsample': 1.0,
            'n_jobs': -1,
            'random_state': 0
        }

    def get_bayesian_space(self) -> dict:
        return {
            "num_leaves": hp.quniform("num_leaves", 8, 64, 1),
            "max_depth": hp.quniform("max_depth", 3, 12, 1),
            "min_child_samples": hp.quniform("min_child_samples", 5, 100, 1),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1)
        }

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
