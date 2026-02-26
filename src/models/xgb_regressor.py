from xgboost import XGBRegressor

from src.enums.objective import Objective
from src.models.xgb_base import XGBBaseWrapper


class XGBRegressorWrapper(XGBBaseWrapper):

    def _get_model_class(self):
        return XGBRegressor

    def get_objective(self) -> Objective:
        return Objective.REGRESSION

    def get_starter_params(self) -> dict:
        return {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1
        }

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba is not supported on regression models")
