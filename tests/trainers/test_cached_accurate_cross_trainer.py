
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from tests.data_load import load_classification_data, load_regression_data
from tests.trainers.trainer_base import TrainerBase


class TestCachedAccurateCrossTrainer(TrainerBase):

    @classmethod
    def setUpClass(cls):
        cls.classification_X, cls.classification_y = load_classification_data()
        cls.regression_X, cls.regression_y = load_regression_data()
        cls.pipeline = HousingPricesCompetitionDTPipeline(cls.regression_X)
        cls.trainer = CachedAccurateCrossTrainer


