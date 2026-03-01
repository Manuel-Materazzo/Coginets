import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score, \
    cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold, GroupKFold

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.objective import Objective
from src.models.model_inference_wrapper import ModelInferenceWrapper
from src.models.model_wrapper import ModelWrapper
from src.pipelines.dt_pipeline import DTPipeline
from abc import ABC, abstractmethod

from src.utils.logger import log


_TARGET_DIR = Path(__file__).resolve().parent.parent.parent / 'target'


def save_model(model: ModelInferenceWrapper):
    _TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with open(_TARGET_DIR / 'model.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model() -> ModelInferenceWrapper:
    with open(_TARGET_DIR / 'model.pkl', 'rb') as file:
        return pickle.load(file)


class Trainer(ABC):
    def __init__(self, pipeline: DTPipeline, model_wrapper: ModelWrapper, metric: AccuracyMetric = AccuracyMetric.MAE,
                 grouping_columns: list[str] = None, n_splits: int = 5):
        self.pipeline: DTPipeline = pipeline
        self.metric: AccuracyMetric = metric
        self.model_wrapper = model_wrapper
        self.grouping_columns = grouping_columns
        self.n_splits = n_splits
        self.evals: list = []

    def get_pipeline(self) -> DTPipeline:
        """
        Returns the pipeline that gets used for training.
        :return:
        """
        return self.pipeline

    def get_model_name(self) -> str:
        return type(self.model_wrapper).__name__

    def get_kfold_type(self) -> any:
        """
        Gets the type of kfold used for training based on model parameters.
        :return:
        """
        if self.model_wrapper.get_objective() == Objective.REGRESSION:
            if self.grouping_columns is not None:
                return GroupKFold(n_splits=self.n_splits)
            else:
                return KFold(n_splits=self.n_splits, random_state=0, shuffle=True)
        elif self.model_wrapper.get_objective() == Objective.CLASSIFICATION:
            if self.grouping_columns is not None:
                return StratifiedGroupKFold(n_splits=self.n_splits, random_state=0, shuffle=True)
            else:
                return StratifiedKFold(n_splits=self.n_splits, random_state=0, shuffle=True)
        else:
            raise ValueError(f"Unsupported objective: {self.model_wrapper.get_objective()}")

    def show_feature_importance(self, X: DataFrame):
        # Apply the same transformations as the training process
        processed_X = self.pipeline.transform(X)

        # Get columns and importance list
        features = list(processed_X.columns)
        importance_df = self.model_wrapper.get_feature_importance(features)

        # Convert to percentage
        total_importance = importance_df['importance'].sum()
        importance_df['importance'] = importance_df['importance'] / total_importance * 100

        log.table(importance_df)
        # plot it!
        plt.figure(figsize=(12, 8))
        plt.xlabel('Importance %')
        plt.ylabel('Feature Type__Name')
        sns.barplot(data=importance_df, x='importance', y='feats')
        plt.show()

    def show_loss(self):
        if len(self.evals) == 0:
            raise ValueError("No model has been fitted with an evaluation set")

        plt.figure(figsize=(12, 6))
        plt.xlabel('Iterations')
        plt.ylabel(self.metric.value)
        plt.title('Loss Over Iterations')

        i = 0

        # add a line to the plot for each training done
        for eval_round in self.evals:
            validation_name = next(iter(eval_round))
            accuracy_metric_name = next(iter(eval_round[validation_name]))
            epochs = len(eval_round[validation_name][accuracy_metric_name])
            x_axis = range(0, epochs)
            plt.plot(x_axis, eval_round[validation_name][accuracy_metric_name], label='Split-{}'.format(i))
            i = i + 1

        plt.legend()
        plt.show()

    def train_model(self, train_X: DataFrame, train_y: Series, val_X: DataFrame = None, val_y: Series = None,
                    iterations=1000, params=None) -> tuple[ModelWrapper, DataFrame | None]:
        """
        Trains a Wrapped Model on the provided training data.
        When validation data is provided, the model is trained with early stopping.
        :param params:
        :param train_X:
        :param train_y:
        :param val_X:
        :param val_y:
        :param iterations:
        :return: Tuple of (trained model wrapper, processed validation data or None).
        """
        params = params or {}
        processed_train_X = self.pipeline.fit_transform(train_X)
        processed_val_X = None

        # if we have validation sets, train with early stopping rounds
        if val_y is not None:
            processed_val_X = self.pipeline.transform(val_X)
            self.model_wrapper.train_until_optimal(processed_train_X, processed_val_X, train_y, val_y, params)
            self.evals.append(self.model_wrapper.get_loss())
        # else train with all the data
        else:
            self.model_wrapper.fit(processed_train_X, train_y, iterations, params)

        return self.model_wrapper, processed_val_X

    @abstractmethod
    def validate_model(self, X: DataFrame, y: Series, log_level=1, iterations=None, params=None,
                       output_prediction_comparison=False) -> tuple[float, int, DataFrame]:
        """
        Validates the model.
        :param X:
        :param y:
        :param log_level:
        :param iterations:
        :param params:
        :param output_prediction_comparison:
        :return:
        """

    def _aggregate_cv_results(self, cv_scores: list, best_rounds: list, oof_comparisons_dataframes: list,
                              log_level: int) -> tuple[float, int, DataFrame]:
        """
        Aggregates cross-validation results: computes mean accuracy, optimal boosting rounds,
        and logs results.
        :param cv_scores: list of accuracy scores from each fold.
        :param best_rounds: list of best iteration counts from each fold.
        :param oof_comparisons_dataframes: list of DataFrames with prediction comparisons per fold.
        :param log_level: verbosity level (0=silent, 1=summary, 2=detailed).
        :return: Tuple of (mean_accuracy, optimal_boost_rounds, oof_prediction_comparisons).
        """
        # compute comparisons across all folds
        if len(oof_comparisons_dataframes) > 0:
            oof_prediction_comparisons = pd.concat(oof_comparisons_dataframes)
        else:
            oof_prediction_comparisons = None

        # Calculate the mean accuracy from cross-validation
        mean_accuracy = np.mean(cv_scores)
        # Calculate optimal boosting rounds
        optimal_boost_rounds = int(np.mean(best_rounds))

        if log_level > 0:
            log.result("Cross-Validation {}".format(self.metric.value), mean_accuracy)
            if log_level > 1:
                log.detail(str(cv_scores))
            log.result("Optimal iterations", optimal_boost_rounds)
            if log_level > 1:
                log.detail(str(best_rounds))

        return mean_accuracy, optimal_boost_rounds, oof_prediction_comparisons

    @staticmethod
    def show_confusion_matrix(real_values: Series, predictions):
        cm = confusion_matrix(real_values, predictions)
        sns.heatmap(cm, annot=True, fmt='', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Real Data')
        plt.show()

    def get_predictions(self, X: DataFrame) -> Series:
        if self.metric == AccuracyMetric.AUC:
            return self.model_wrapper.predict_proba(X)
        else:
            return self.model_wrapper.predict(X)

    def calculate_accuracy(self, predictions: Series, real_values: Series) -> float:
        """
        Calculates the accuracy of the provided predictions, using the metric specified when creating the trainer.
        :param predictions:
        :param real_values:
        :return:
        """
        match self.metric:
            case AccuracyMetric.MAE:
                return mean_absolute_error(real_values, predictions)
            case AccuracyMetric.MSE:
                return mean_squared_error(real_values, predictions)
            case AccuracyMetric.RMSE:
                return math.sqrt(mean_squared_error(real_values, predictions))
            case AccuracyMetric.AUC:
                return roc_auc_score(real_values, predictions)
            case AccuracyMetric.Accuracy:
                return accuracy_score(real_values, predictions)
            case AccuracyMetric.QWK:
                return cohen_kappa_score(real_values, predictions, weights='quadratic')
            case _:
                raise ValueError(f"Unknown accuracy metric: {self.metric}")
