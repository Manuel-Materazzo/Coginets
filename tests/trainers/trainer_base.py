import inspect
import unittest
import importlib
import pkgutil
import src.models as models

from src.enums.accuracy_metric import AccuracyMetric
from tests.data_load import load_classification_data, load_regression_data
import matplotlib.pyplot as plt


class TrainerBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = None
        cls.trainer = None

    @classmethod
    def add_test_method(cls, name, func, **kwargs):
        def method(self):
            func(self, **kwargs)

        method.__name__ = name
        setattr(cls, name, method)

    def _validate(self, model_type, metric, X, y):
        if self.trainer is not None:
            trainer_instance = self._instantiate_trainer(X, y, model_type, metric)

            accuracy, best_iteration, prediction_comparison = trainer_instance.validate_model(X, y, log_level=0,
                                                                                              output_prediction_comparison=True)

            # disable plot output
            plt.switch_backend("Agg")
            plt.ioff()

            trainer_instance.show_loss()
            trainer_instance.show_feature_importance(X)
            self.assertGreaterEqual(accuracy, 0)
            self.assertGreaterEqual(best_iteration, 0)
            self.assertFalse(prediction_comparison.empty)

    def _train(self, model_type, metric, X, y):
        if self.trainer is not None:
            trainer_instance = self._instantiate_trainer(X, y, model_type, metric)

            model = trainer_instance.train_model(X, y, iterations=10)
            self.assertIsNotNone(model)

    def _instantiate_trainer(self, X, y, model_type, metric):
        sig = inspect.signature(self.trainer.__init__)
        data_required = 'X' in sig.parameters

        if data_required:
            return self.trainer(pipeline=self.pipeline, X=X, y=y, model_wrapper=model_type,
                                metric=metric, n_splits=2)
        else:
            return self.trainer(pipeline=self.pipeline, model_wrapper=model_type, metric=metric, n_splits=2)


classification_X, classification_y = load_classification_data()
regression_X, regression_y = load_regression_data()


def add_classification_tests(model_name, model):
    validation_test_name = f'test_validate_{model_name.lower()}'
    training_test_name = f'test_train_{model_name.lower()}'

    TrainerBase.add_test_method(
        validation_test_name,
        TrainerBase._validate,
        model_type=model(early_stopping_rounds=1),
        metric=AccuracyMetric.AUC,
        X=classification_X,
        y=classification_y
    )

    TrainerBase.add_test_method(
        training_test_name,
        TrainerBase._train,
        model_type=model(early_stopping_rounds=1),
        metric=AccuracyMetric.AUC,
        X=classification_X,
        y=classification_y
    )


def add_regression_tests(model_name, model):
    validation_test_name = f'test_validate_{model_name.lower()}'
    training_test_name = f'test_train_{model_name.lower()}'

    TrainerBase.add_test_method(
        validation_test_name,
        TrainerBase._validate,
        model_type=model(early_stopping_rounds=1),
        metric=AccuracyMetric.RMSE,
        X=regression_X,
        y=regression_y
    )
    TrainerBase.add_test_method(
        training_test_name,
        TrainerBase._train,
        model_type=model(early_stopping_rounds=1),
        metric=AccuracyMetric.RMSE,
        X=regression_X,
        y=regression_y
    )


# Discover modules dynamically
for loader, module_name, is_pkg in pkgutil.iter_modules(models.__path__):
    # import model module
    module = importlib.import_module(f'src.models.{module_name}')
    # instantiate tests for the model
    for class_name, obj in inspect.getmembers(module, inspect.isclass):
        if class_name.endswith('ClassifierWrapper'):
            add_classification_tests(class_name, obj)
        elif class_name.endswith('RegressorWrapper'):
            add_regression_tests(class_name, obj)
