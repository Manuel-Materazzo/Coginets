from xgboost import XGBClassifier

from src.enums.objective import Objective
from src.models.xgb_base import XGBBaseWrapper


class XGBClassifierWrapper(XGBBaseWrapper):

    def _get_model_class(self):
        return XGBClassifier

    def get_objective(self) -> Objective:
        return Objective.CLASSIFICATION

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

    def predict_proba(self, X) -> any:
        return self.model.predict_proba(X)[:, 1]
