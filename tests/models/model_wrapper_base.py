import unittest

from sklearn.model_selection import train_test_split

from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline


class ModelWrapperBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = None
        cls.y = None
        cls.model = None
        cls.base_model = None
        cls.objective = None

    def test_objective(self):
        if self.model is not None:
            self.assertEqual(self.model.get_objective(), self.objective)

    def test_base_model(self):
        if self.model is not None:
            self.assertIsInstance(self.model.get_base_model(10, {}), self.base_model)

    def test_starter_params(self):
        if self.model is not None:
            self.assertTrue(self.model.get_starter_params())

    def test_grid_space(self):
        if self.model is not None:
            self.assertTrue(self.model.get_grid_space())

    def test_bayesian_space(self):
        if self.model is not None:
            self.assertTrue(self.model.get_bayesian_space())

    def test_unfit_statistical_methods(self):
        if self.model is not None:
            self.model.model = None
            self.model.importances = None
            with self.assertRaises((ValueError, RuntimeError)):
                self.model.get_loss()
            with self.assertRaises((ValueError, RuntimeError)):
                self.model.get_feature_importance(self.X)

    def test_optimal_fit_and_statistical_methods(self):
        if self.model is not None:
            train_X, val_X, train_y, val_y = self._train_test_split()

            self.model.train_until_optimal(train_X, val_X, train_y, val_y)

            self.assertGreaterEqual(self.model.get_best_iteration(), 0)
            self.assertIsNotNone(self.model.get_loss())
            self.assertIsNotNone(self.model.get_feature_importance(self.X))

    def test_fit_and_predict(self):
        if self.model is not None:
            pipeline = HousingPricesCompetitionDTPipeline(self.X)
            train_X = pipeline.fit_transform(self.X)

            self.model.fit(train_X, self.y, 10)
            predictions = self.model.predict(train_X)

            self.assertIsNotNone(predictions)

    def _train_test_split(self):
        pipeline = HousingPricesCompetitionDTPipeline(self.X)

        train_X, val_X, train_y, val_y = train_test_split(self.X, self.y, random_state=0)
        train_X = pipeline.fit_transform(train_X)
        val_X = pipeline.transform(val_X)

        return train_X, val_X, train_y, val_y
