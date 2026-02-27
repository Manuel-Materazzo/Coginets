from sklearn.ensemble import HistGradientBoostingClassifier

from src.enums.objective import Objective
from src.models.hgb_classifier import HGBClassifierWrapper
from tests.data_load import load_classification_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestHgbCatboostClassifier(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_classification_data()
        cls.model = HGBClassifierWrapper(early_stopping_rounds=1, permutation_importance_repeats=2)
        cls.base_model = HistGradientBoostingClassifier
        cls.objective = Objective.CLASSIFICATION
