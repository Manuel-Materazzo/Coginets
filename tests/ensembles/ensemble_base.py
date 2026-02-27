import unittest
import importlib
import pkgutil

import src.models as models

from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from tests.data_load import load_regression_data
from tests.dynamic_modules import get_local_classes
import matplotlib.pyplot as plt


class EnsembleBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ensemble = None

    @classmethod
    def add_test_method(cls, name, func, **kwargs):
        def method(self):
            func(self, **kwargs)

        method.__name__ = name
        setattr(cls, name, method)

    def _validate_and_show_weights(self, members, X, y):
        if self.ensemble is not None:
            ensemble_instance = self.ensemble(members=members)
            ensemble_instance.trials = 3  # lower max trials to speed up testing
            accuracy = ensemble_instance.validate_models_and_show_leaderboard(X, y)

            # disable plot output
            plt.switch_backend("Agg")
            plt.ioff()

            ensemble_instance.show_weights()

            self.assertGreaterEqual(accuracy, 0)

    def _fit_and_predict(self, members, X, y):
        if self.ensemble is not None:
            ensemble_instance = self.ensemble(members=members)
            ensemble_instance.trials = 3  # lower max trials to speed up testing
            ensemble_instance.train(X, y)
            pred_X = pipeline.transform(X)
            predictions = ensemble_instance.predict(pred_X)
            self.assertIsNotNone(predictions)


regression_X, regression_y = load_regression_data()
pipeline = HousingPricesCompetitionDTPipeline(regression_X)

models_modules = []

# Discover model modules dynamically
for _, model_module_name, _ in pkgutil.iter_modules(models.__path__):
    # import model module
    model_module = importlib.import_module(f'src.models.{model_module_name}')
    if model_module_name.endswith('_regressor'):
        models_modules.append(model_module)

# create consecutive pairs so each model is tested at least once, without full cartesian product
combinations = list(zip(models_modules, models_modules[1:] + [models_modules[0]]))

# create a test case for each model combination
for model_module_1, model_module_2 in combinations:

    # we iterate through the classes of the module, because someone could put more than one, for some godforsaken reason
    for model_name_1, model_obj_1 in get_local_classes(model_module_1):
        for model_name_2, model_obj_2 in get_local_classes(model_module_2):
            validate_test_name = f'test_validate_and_show_weights_{model_name_1.lower()}_{model_name_2.lower()}'
            predict_test_name = f'test_fit_and_predict_{model_name_1.lower()}_{model_name_2.lower()}'

            # instantiate model
            model_instance_1 = model_obj_1(early_stopping_rounds=1)
            model_instance_2 = model_obj_2(early_stopping_rounds=1)

            members_list = [
                {
                    'trainer': CachedAccurateCrossTrainer(pipeline, model_instance_1, regression_X, regression_y,
                                                         n_splits=2),
                    'params': model_instance_1.get_starter_params(),
                    'optimizer': None
                },
                {
                    'trainer': CachedAccurateCrossTrainer(pipeline, model_instance_2, regression_X, regression_y,
                                                         n_splits=2),
                    'params': model_instance_2.get_starter_params(),
                    'optimizer': None
                }
            ]

            # add dynamic tests
            EnsembleBase.add_test_method(
                validate_test_name,
                EnsembleBase._validate_and_show_weights,
                members=members_list,
                X=regression_X,
                y=regression_y,
            )

            EnsembleBase.add_test_method(
                predict_test_name,
                EnsembleBase._fit_and_predict,
                members=members_list,
                X=regression_X,
                y=regression_y,
            )
