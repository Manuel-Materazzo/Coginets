import optuna
import platform
import optuna_distributed
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pandas import DataFrame, Series

from src.ensembles.ensemble import EnsembleMember, Ensemble
from src.utils.logger import log


class WeightedEnsemble(Ensemble):

    def __init__(self, members: list[EnsembleMember]):
        super().__init__(members)
        self.trials = 4000

    def post_validation_callback(self, X: DataFrame, y: Series, oof_predictions: DataFrame):
        """
        Callback to optimize weights when done training. Do not call manually.
        :param oof_predictions:
        :param X:
        :param y:
        :return:
        """
        log.info("Optimizing ensemble weights...")
        self._optimize_weights(oof_predictions, y)

    def show_weights(self):
        """
        Shows ensemble weights
        :return:
        """

        if self.weights is None:
            log.warning("Weights are still not optimized. This probably means the validation was skipped or the callback is set incorrectly.")
            return

        # Pie chart
        plt.pie(self.weights, labels=self.weights.index, autopct="%.0f%%")
        plt.title('Ensemble weights')
        plt.show()

    def _optimize_weights(self, oof_predictions: DataFrame, y: Series):
        """
        Trains each model of the ensemble on half of the provided data, and calculates the optimal ensemble weights.
        :param oof_predictions:
        :param y:
        :return:
        """

        # extract arrays from predictions dataframe
        model_names = oof_predictions.columns
        predictions_array = [oof_predictions[col].values for col in model_names]

        # optimize weights to maximize prediction accuracy
        raw_weights = self._optuna_weight_study(y, predictions_array)
        self.weights: Series = Series(raw_weights, index=model_names)

    def _optuna_weight_study(self, real_values: Series, predictions_array: list) -> list:
        """
        Uses Optuna to optimize the weights of the ensemble and returns them.
        :param real_values:
        :param predictions_array:
        :return:
        """
        # set log level
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        optuna_distributed.config.disable_logging()
        # create a study
        sampler = optuna.samplers.CmaEsSampler(seed=0)
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptWeights", direction='minimize')
        # leverage distributed training on linux
        if platform.system() != 'Windows':
            study = optuna_distributed.from_study(study)
        # define an objective and start the study
        objective_partial = partial(self._objective, real_values=real_values, predictions_array=predictions_array)
        study.optimize(objective_partial, n_trials=self.trials, n_jobs=-1)
        # get weights
        return [study.best_params[f"weight{n}"] for n in range(len(predictions_array))]

    def _objective(self, trial, real_values, predictions_array):
        """
        Optimization objective for optuna.
        :param trial:
        :param real_values:
        :param predictions_array:
        :return:
        """
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-14, 1) for n in range(len(predictions_array))]

        # Calculate the weighted prediction
        predictions = np.average(np.array(predictions_array).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = self.members[0]['trainer'].calculate_accuracy(predictions, real_values)
        return score

    def predict(self, X: DataFrame) -> Series:
        if len(self.weights) == 0:
            log.warning("Weights are still not optimized. This probably means the validation was skipped or the callback is set incorrectly.")
            return Series([])

        if len(self.models) == 0:
            log.error("No models trained, use train() first")
            return Series([])

        predictions_array = []

        # for each trained model
        for model in self.models:
            # make prediction and add to the prediction list
            predictions_array.append(model.predict(X))

        weighted_pred = np.average(np.array(predictions_array).T, axis=1, weights=self.weights)
        return weighted_pred
