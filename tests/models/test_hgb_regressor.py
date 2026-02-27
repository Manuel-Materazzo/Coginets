
from sklearn.ensemble import HistGradientBoostingRegressor

from src.enums.objective import Objective
from src.models.hgb_regressor import HGBRegressorWrapper
from tests.data_load import load_regression_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestHgbRegresor(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()
        cls.model = HGBRegressorWrapper(early_stopping_rounds=1, permutation_importance_repeats=2)
        cls.base_model = HistGradientBoostingRegressor
        cls.objective = Objective.REGRESSION
