
from src.enums.accuracy_metric import AccuracyMetric
from src.models.xgb_classifier import XGBClassifierWrapper
from src.models.xgb_regressor import XGBRegressorWrapper
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

    def test_validate_grouped_regression(self):
        X = self.regression_X.copy()
        X['_group'] = range(len(X))
        pipeline = HousingPricesCompetitionDTPipeline(X)
        model = XGBRegressorWrapper(early_stopping_rounds=1)
        trainer = CachedAccurateCrossTrainer(
            pipeline=pipeline, model_wrapper=model, X=X, y=self.regression_y,
            metric=AccuracyMetric.RMSE, grouping_columns=['_group'], n_splits=2
        )
        accuracy, best_iteration, _ = trainer.validate_model(X, self.regression_y, log_level=0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(best_iteration, 0)

    def test_validate_grouped_classification(self):
        X = self.classification_X.copy()
        X['_group'] = range(len(X))
        pipeline = HousingPricesCompetitionDTPipeline(X)
        model = XGBClassifierWrapper(early_stopping_rounds=1)
        trainer = CachedAccurateCrossTrainer(
            pipeline=pipeline, model_wrapper=model, X=X, y=self.classification_y,
            metric=AccuracyMetric.AUC, grouping_columns=['_group'], n_splits=2
        )
        accuracy, best_iteration, _ = trainer.validate_model(X, self.classification_y, log_level=0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(best_iteration, 0)


