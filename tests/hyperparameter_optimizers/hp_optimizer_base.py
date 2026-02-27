import importlib
import pkgutil
import unittest
import src.models as models

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.trainers.simple_trainer import SimpleTrainer
from tests.data_load import load_classification_data, load_regression_data

from tests.dynamic_modules import get_local_classes, is_x_in_signature, lobotomize_grid_space


class HpOptimizerBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = None
        cls.optimizer = None

    @classmethod
    def add_test_method(cls, name, func, **kwargs):
        def method(self):
            func(self, **kwargs)

        method.__name__ = name
        setattr(cls, name, method)

    def _tune(self, trainer, model, X, y, metric, direction):
        if self.optimizer is not None:
            trainer_instance = self._instantiate_trainer(trainer, X, y, model, metric)
            optimizer_instance = self.optimizer(trainer_instance, model, direction=direction)
            optimizer_instance.trials = 2

            optimized_params = optimizer_instance.tune(X, y, 0.1)

            self.assertIsNotNone(optimized_params)

    def _instantiate_trainer(self, trainer, X, y, model_type, metric):
        if is_x_in_signature(trainer):
            return trainer(pipeline=self.pipeline, X=X, y=y, model_wrapper=model_type,
                           metric=metric, n_splits=2)
        else:
            return trainer(pipeline=self.pipeline, model_wrapper=model_type, metric=metric, n_splits=2)


# load data
classification_X, classification_y = load_classification_data()
regression_X, regression_y = load_regression_data()

# Discover model modules dynamically
models_modules = []

# Discover model modules dynamically
for _, model_module_name, _ in pkgutil.iter_modules(models.__path__):
    # import model module
    model_module = importlib.import_module(f'src.models.{model_module_name}')
    if model_module_name.endswith('_classifier') or model_module_name.endswith('_regressor'):
        models_modules.append(model_module)

# create a test case for each model using SimpleTrainer
# trainer-model compatibility is already covered by trainer tests
for model_module in models_modules:

    for model_name, model_obj in get_local_classes(model_module):

        test_name = f'test_tune_{model_name.lower()}_simpletrainer'

        # instantiate model object and reduce grid space to avoid wasting time
        model_instance = model_obj(early_stopping_rounds=1)
        lobotomize_grid_space(model_instance)

        # of course, classifiers and regressors have different test cases
        if model_name.endswith('ClassifierWrapper'):
            HpOptimizerBase.add_test_method(
                test_name,
                HpOptimizerBase._tune,
                trainer=SimpleTrainer,
                model=model_instance,
                metric=AccuracyMetric.AUC,
                direction=OptimizationDirection.MAXIMIZE,
                X=classification_X,
                y=classification_y,
            )
        elif model_name.endswith('RegressorWrapper'):
            HpOptimizerBase.add_test_method(
                test_name,
                HpOptimizerBase._tune,
                trainer=SimpleTrainer,
                model=model_instance,
                metric=AccuracyMetric.RMSE,
                direction=OptimizationDirection.MINIMIZE,
                X=regression_X,
                y=regression_y
            )
