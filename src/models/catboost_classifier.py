from catboost import CatBoostClassifier

from src.enums.objective import Objective
from src.models.catboost_base import CatBoostBaseWrapper


class CatBoostClassifierWrapper(CatBoostBaseWrapper):

    def _get_model_class(self):
        return CatBoostClassifier

    def get_objective(self) -> Objective:
        return Objective.CLASSIFICATION

    def get_starter_params(self) -> dict:
        return {
            'loss_function': 'Logloss',
            'grow_policy': 'SymmetricTree',
            'bagging_temperature': 0.50,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 1.25,
            'min_data_in_leaf': 24,
            'thread_count': -1
        }

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
