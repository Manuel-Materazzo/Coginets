from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score, make_scorer

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer
from src.utils.logger import log


class DefaultGridOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer, model_wrapper: ModelWrapper,
                 direction: OptimizationDirection = OptimizationDirection.MINIMIZE):
        super().__init__(trainer, model_wrapper, direction=direction)

    def __get_full_pipeline(self, optimal_boosting_rounds: int) -> Pipeline:
        """
        Gets a pipeline that includes model training
        :param optimal_boosting_rounds:
        :return:
        """
        base_model = self.trainer.model_wrapper.get_base_model(optimal_boosting_rounds, self.params.copy())
        return self.trainer.get_pipeline().get_pipeline_with_training(base_model)

    def tune(self, X: DataFrame, y: Series, final_lr: float, log_level=0) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a grid search
        Trains a cross-validated model for each combination of hyperparameters, and picks the best based on MAE
        :param log_level:
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        # get optimal boost rounds
        optimal_br = self.get_optimal_boost_rounds(X, y) or 1
        index = 1

        # get a list of spaces to optimize using sequential steps
        for step_space in self.model_wrapper.get_grid_space():

            step_space = step_space.copy()

            # recalibrate iteration if needed
            if step_space['recalibrate_iterations']:
                optimal_br = self.get_optimal_boost_rounds(X, y) or 1
            # avoid to pass useless arguments to the model
            del step_space['recalibrate_iterations']

            # add 'model__' to every param, to adapt to the model training pipeline terminology
            step_space = {'model__' + key: value for key, value in step_space.items()}

            log.info("Step {}".format(index))
            # grid search for best params
            optimal_params = self.__do_grid_search(self.__get_full_pipeline(optimal_br), X, y, step_space, log_level)

            # remove 'model__' from every param in order to have clean values
            fixed_optimal_params = {key[7:]: value for key, value in optimal_params.items() if
                                    key.startswith('model__')}

            # update defaults with new optimal params
            self.params.update(fixed_optimal_params)
            index += 1

        self.params['learning_rate'] = final_lr

        return self.params

    def __do_grid_search(self, pipeline: Pipeline, X: DataFrame, y: Series, param_grid: dict, log_level=1) -> dict:
        """
        Trains cross-validated model for each combination of the provided hyperparameters, and picks the best based on MAE
        :param pipeline:
        :param X:
        :param y:
        :param param_grid:
        :return:
        """

        match self.trainer.metric:
            case AccuracyMetric.MAE:
                scoring = 'neg_mean_absolute_error'
            case AccuracyMetric.MSE:
                scoring = 'neg_mean_squared_error'
            case AccuracyMetric.RMSE:
                scoring = 'neg_root_mean_squared_error'
            case AccuracyMetric.AUC:
                scoring = 'roc_auc'
            case AccuracyMetric.QWK:
                # custom scorer for quadratic weighted kappa
                scoring = make_scorer(cohen_kappa_score, weights="quadratic")
            case _:
                scoring = self.trainer.metric.value.lower()

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        if log_level > 0:
            log.result("Best parameters found", grid_search.best_params_)
            log.result("Best accuracy", -grid_search.best_score_)

        if log_level > 1:
            # Print all parameters and corresponding MAE
            results = grid_search.cv_results_
            for i in range(len(results['params'])):
                log.detail(f"Parameters: {results['params'][i]}")
                log.detail(f"Accuracy: {abs(results['mean_test_score'][i])}")

        return grid_search.best_params_
