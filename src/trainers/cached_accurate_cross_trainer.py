import pandas as pd
from pandas import DataFrame, Series

from src.enums.accuracy_metric import AccuracyMetric
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.trainer import Trainer


class CachedAccurateCrossTrainer(Trainer):
    """
    Wrapper of AccurateCrossTrainer that takes X and Y at initialization time and caches kfold splits.
    """

    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, X: DataFrame, y: Series,
                 metric: AccuracyMetric = AccuracyMetric.MAE, grouping_columns: list[str] = None,
                 n_splits: int = 5):
        super().__init__(pipeline, model_wrapper, metric=metric, grouping_columns=grouping_columns, n_splits=n_splits)
        self.X = X
        self.y = y
        self.splits = self.__cache_splits()
        self.trainer = AccurateCrossTrainer(pipeline, model_wrapper, metric=metric, grouping_columns=grouping_columns,
                                            n_splits=n_splits)

    def __cache_splits(self) -> list:
        """
        Splits X and y into 5 folds at initialization time and returns them as a list.
        :return:
        """
        kf = self.get_kfold_type()

        splits = []

        if self.grouping_columns is not None:
            groups = self.X[self.grouping_columns]
        else:
            groups = None

        for train_index, val_index in kf.split(self.X, self.y, groups):
            # split train and validation data
            train_X, val_X = self.X.iloc[train_index], self.X.iloc[val_index]
            train_y, val_y = self.y.iloc[train_index], self.y.iloc[val_index]

            splits.append([train_X, val_X, train_y, val_y])

        return splits

    def __cross_train(self, split, iterations=None, params=None,
                      output_prediction_comparison=False) -> tuple[int, float, DataFrame]:

        # if no rounds, train with early stopping
        if iterations is None:
            _, processed_val_X = self.trainer.train_model(split[0], split[2], split[1], split[3], params=params)
        # else train normally
        else:
            self.trainer.train_model(split[0], split[2], iterations=iterations, params=params)
            processed_val_X = self.trainer.pipeline.transform(split[1])

        # Predict and calculate accuracy
        predictions = self.get_predictions(processed_val_X)
        accuracy = self.calculate_accuracy(predictions, split[3])

        # create a dataframe with comparison
        if output_prediction_comparison:
            prediction_comparison = pd.DataFrame({'real_values': split[3], 'predictions': list(predictions)})
        else:
            prediction_comparison = None

        try:
            # number of boosting rounds used in the best model, accuracy
            return self.model_wrapper.get_best_iteration(), accuracy, prediction_comparison
        # if the model was trained without early stopping, return the provided training rounds
        except AttributeError:
            return iterations, accuracy, prediction_comparison

    def validate_model(self, X: DataFrame, y: Series, log_level=2, iterations=None, params=None,
                       output_prediction_comparison=False) -> tuple[float, int, DataFrame]:
        """
        Trains 5 Models on the provided training data by cross-validation using cached splits.
        Data is splitted into 5 folds, each model is trained on 4 folds and validated on 1 fold.
        The validation fold is always different, so we are basically training and validating over the entire dataset.
        Accuracy score and optimal iterations of each model are then meaned to get overall values.
        If no rounds are provided, the models are trained using early stopping and will return the optimal number of
        boosting rounds alongside the Accuracy.

        X and y must match the data provided at initialization time, as cached splits are used.

        :param output_prediction_comparison: whether to output a dataframe containing predictions and actual values.
        :param X:
        :param y:
        :param iterations:
        :param log_level:
        :param params:
        :return:
        """
        if not X.equals(self.X) or not y.equals(self.y):
            raise ValueError("X and y must match the data provided at initialization time, "
                             "as this trainer uses cached splits.")

        # Placeholder for cross-validation accuracy scores
        cv_scores = []
        best_rounds = []

        self.evals = []
        oof_comparisons_dataframes = []

        # Loop through each fold
        for split in self.splits:
            # cross train
            best_iteration, accuracy, oof_prediction_comparison = self.__cross_train(split, iterations=iterations,
                                                                                     params=params,
                                                                                     output_prediction_comparison=output_prediction_comparison)

            # when oof prediction save is enabled, add the prediction to the list
            if oof_prediction_comparison is not None:
                oof_comparisons_dataframes.append(oof_prediction_comparison)

            # add the best iteration and accuracy to lists
            best_rounds.append(best_iteration or 0)
            cv_scores.append(accuracy)

        # compute comparisons across all folds
        if len(oof_comparisons_dataframes) > 0:
            pd.concat(oof_comparisons_dataframes)

        # extract evals
        self.evals = self.trainer.evals

        return self._aggregate_cv_results(cv_scores, best_rounds, oof_comparisons_dataframes, log_level)
