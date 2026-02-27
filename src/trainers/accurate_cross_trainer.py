import pandas as pd
from pandas import DataFrame, Series

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline
from src.trainers.trainer import Trainer


class AccurateCrossTrainer(Trainer):

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, metric: AccuracyMetric = AccuracyMetric.MAE,
                 grouping_columns: list[str] = None, n_splits: int = 5):
        super().__init__(pipeline, model_wrapper, metric=metric, grouping_columns=grouping_columns, n_splits=n_splits)

    def __cross_train(self, X: DataFrame, y: Series, train_index: int, val_index: int, iterations=None,
                      params=None, output_prediction_comparison=False) -> (int, int):
        """
        Trains a Model on the provided training data by splitting it into training and validation sets.
        The split function does not shuffle, and it's based on the provided indexes.
        If no rounds are provided, the model is trained using early stopping and will return the optimal number of
        iterations alongside the accuracy.
        :param X:
        :param y:
        :param train_index:
        :param val_index:
        :param iterations:
        :param params:
        :return:
        """
        # split train and validation data
        train_X, val_X = X.iloc[train_index], X.iloc[val_index]
        train_y, val_y = y.iloc[train_index], y.iloc[val_index]

        # if no rounds, train with early stopping
        if iterations is None:
            self.train_model(train_X, train_y, val_X, val_y, params=params)
        # else train normally
        else:
            self.train_model(train_X, train_y, iterations=iterations, params=params)

        # re-process val_X to obtain accuracy
        processed_val_X = self.pipeline.transform(val_X)

        # Predict and calculate accuracy
        predictions = self.get_predictions(processed_val_X)
        accuracy = self.calculate_accuracy(predictions, val_y)

        # create a dataframe with comparison
        if output_prediction_comparison:
            prediction_comparison = pd.DataFrame({'real_values': val_y, 'predictions': list(predictions)})
        else:
            prediction_comparison = None

        try:
            # number of boosting rounds used in the best model, accuracy
            return self.model_wrapper.get_best_iteration(), accuracy, prediction_comparison
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return iterations, accuracy, prediction_comparison

    def validate_model(self, X: DataFrame, y: Series, log_level=2, iterations=None, params=None,
                       output_prediction_comparison=False) -> (float, int, DataFrame):
        """
        Trains 5 Models on the provided training data by cross-validation.
        Data is splitted into 5 folds, each model is trained on 4 folds and validated on 1 fold.
        The validation fold is always different, so we are basically training and validating over the entire dataset.
        Accuracy score and optimal iterations of each model are then meaned to get overall values.
        If no rounds are provided, the models are trained using early stopping and will return the optimal number of
        boosting rounds alongside the Accuracy.

        :param output_prediction_comparison: whether to output a dataframe containing predictions and actual values.
        :param X:
        :param y:
        :param iterations:
        :param log_level:
        :param params:
        :return:
        """
        # Initialize KFold
        kf = self.get_kfold_type()

        if self.grouping_columns is not None:
            groups = X[self.grouping_columns]
        else:
            groups = None

        self.evals = []
        oof_comparisons_dataframes = []

        # Placeholder for cross-validation MAE scores
        cv_scores = []
        best_rounds = []

        # Loop through each fold
        for train_index, val_index in kf.split(X, y, groups):
            best_iteration, accuracy, oof_prediction_comparison = self.__cross_train(X, y, train_index, val_index,
                                                                                     iterations=iterations,
                                                                                     params=params,
                                                                                     output_prediction_comparison=output_prediction_comparison)
            # when oof prediction save is enabled, add the prediction to the list
            if oof_prediction_comparison is not None:
                oof_comparisons_dataframes.append(oof_prediction_comparison)

            # add the best iteration and accuracy to lists
            best_rounds.append(best_iteration or 0)
            cv_scores.append(accuracy)

        return self._aggregate_cv_results(cv_scores, best_rounds, oof_comparisons_dataframes,
                                             log_level, iterations, params, X, y)
