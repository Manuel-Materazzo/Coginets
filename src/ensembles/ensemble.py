from abc import abstractmethod
from typing import TypedDict

import numpy as np
from pandas import DataFrame, Series

from src.enums.accuracy_metric import AccuracyMetric
from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.models.model_inference_wrapper import ModelInferenceWrapper
from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer
from src.utils.logger import log

EnsembleMember = TypedDict('EnsembleMember',
                           {'trainer': Trainer, 'params': dict | None, 'optimizer': HyperparameterOptimizer | None})

LeaderboardEntry = TypedDict('LeaderboardEntry', {'model_name': str, 'accuracy': float, 'iterations': int})


class Ensemble(ModelInferenceWrapper):

    def __init__(self, members: list[EnsembleMember]):
        self.members = members
        self.accuracy_metric = None
        self.leaderboard = None
        self.weights = None
        self.oof_predictions = None
        self.models: list[ModelWrapper] = []

    def validate_models_and_show_leaderboard(self, X: DataFrame, y: Series) -> float:
        """
        Trains each model of the ensemble and optimizes params if needed.
        Computes an individual model leaderboard, and returns the mean accuracy value across all models.
        :param X:
        :param y:
        :return:
        """
        # if we didn't compute a leaderboard before
        if self.leaderboard is None:
            leaderboard_list: list[LeaderboardEntry] = []
            self.oof_predictions = DataFrame()
            # train each model in the ensemble
            i = 0
            for member in self.members:
                # get the trainer and the params
                trainer = member['trainer']
                params = member['params']
                optimizer = member['optimizer']

                # check accuracy metric consistency
                if self.accuracy_metric is None:
                    self.accuracy_metric: AccuracyMetric = trainer.metric
                if self.accuracy_metric != trainer.metric:
                    log.warning("Accuracy metric is different across trainers")

                with log.group(trainer.get_model_name()):

                    # if we have an optimizer set and no params are provided, calculate optimal params
                    if optimizer is not None and params is None:
                        log.info("No hyperparams provided, auto-optimizing...")
                        params = optimizer.tune(X, y, 0.03)
                        # save optimized params for later (full training)
                        member['params'] = params
                        log.success("Optimal hyperparams found")

                    # train model
                    accuracy, iterations, prediction_comparisons = trainer.validate_model(X, y, log_level=0, params=params,
                                                                                          output_prediction_comparison=True)
                    log.result("Accuracy", accuracy)
                    log.result("Iterations", iterations)

                    # append model results to the leaderboard
                    leaderboard_list.append(
                        LeaderboardEntry(model_name=trainer.get_model_name(), accuracy=accuracy, iterations=iterations)
                    )

                    # append predictions to the out of fold predictions dataframe
                    self.oof_predictions[trainer.get_model_name() + '_' + str(i)] = prediction_comparisons['predictions']
                    i += 1

            self.leaderboard: DataFrame = DataFrame.from_records(leaderboard_list)
            self.leaderboard.sort_values('accuracy', ascending=True, inplace=True)

        # prints the leaderboard
        log.section("Leaderboard")
        log.table(self.leaderboard)
        log.end_section()
        self.oof_predictions.sort_index(inplace=True)

        # do the callback, in order to execute whatever post-processing is needed by child classes
        self.post_validation_callback(X, y, self.oof_predictions)

        # return mean accuracy
        return np.mean(self.leaderboard['accuracy'])

    def train(self, X: DataFrame, y: Series) -> ModelInferenceWrapper:
        """
        Trains each model of the ensemble on the whole data.
        :param X:
        :param y:
        :return:
        """

        if self.leaderboard is None:
            log.info("Evaluating and optimizing models...")
            self.validate_models_and_show_leaderboard(X, y)

        # train each model in the ensemble
        for member in self.members:
            # get the trainer and the params
            trainer = member['trainer']
            params = member['params']

            log.info("Training {}...".format(trainer.get_model_name()))

            # get iterations from leaderboard
            iterations = \
                self.leaderboard.loc[self.leaderboard['model_name'] == trainer.get_model_name()].iterations.values[0]
            # train model
            model, _ = trainer.train_model(X, y, iterations=iterations, params=params)
            self.models.append(model)

        return self

    @abstractmethod
    def predict(self, X: DataFrame) -> Series:
        """
        Predicts the target for the given input data.
        :param X:
        :return:
        """

    @abstractmethod
    def post_validation_callback(self, X: DataFrame, y: Series, oof_predictions: DataFrame):
        """
        Callback when done validating.
        :param oof_predictions:
        :param X:
        :param y:
        :return:
        """
