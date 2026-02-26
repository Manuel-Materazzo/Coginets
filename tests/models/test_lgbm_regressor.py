from lightgbm import LGBMRegressor

from src.enums.objective import Objective
from src.models.lgbm_regressor import LGBMRegressorWrapper
from tests.data_load import load_regression_data
from tests.models.model_wrapper_base import ModelWrapperBase


class TestLgmbRegresor(ModelWrapperBase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()
        cls.model = LGBMRegressorWrapper(early_stopping_rounds=1)
        cls.base_model = LGBMRegressor
        cls.objective = Objective.REGRESSION
