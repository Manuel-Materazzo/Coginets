from catboost import CatBoostRegressor

from src.enums.objective import Objective
from src.models.catboost_base import CatBoostBaseWrapper


class CatBoostRegressorWrapper(CatBoostBaseWrapper):

    def _get_model_class(self):
        return CatBoostRegressor

    def get_objective(self) -> Objective:
        return Objective.REGRESSION

    def get_starter_params(self) -> dict:
        return {
            'loss_function': 'RMSE',
            'grow_policy': 'SymmetricTree',
            'bagging_temperature': 0.50,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 1.25,
            'min_data_in_leaf': 24,
            'thread_count': -1
        }

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba is not supported on regression models")
