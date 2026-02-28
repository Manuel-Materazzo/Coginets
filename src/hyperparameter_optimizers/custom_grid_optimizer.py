import itertools

from pandas import DataFrame, Series

from src.enums.optimization_direction import OptimizationDirection
from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer
from src.utils.logger import log


class CustomGridOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer, model_wrapper: ModelWrapper,
                 direction: OptimizationDirection = OptimizationDirection.MINIMIZE):
        super().__init__(trainer, model_wrapper, direction=direction)

    def tune(self, X: DataFrame, y: Series, final_lr: float) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a grid search
        Trains a cross-validated model for each combination of hyperparameters, and picks the best based on MAE
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        # get optimal boost rounds
        optimal_br = self.get_optimal_boost_rounds(X, y)

        index = 1

        # get a list of spaces to optimize using sequential steps
        for step_space in self.model_wrapper.get_grid_space():

            step_space = step_space.copy()

            # recalibrate iteration if needed
            if step_space['recalibrate_iterations']:
                optimal_br = self.get_optimal_boost_rounds(X, y)
            # avoid to pass useless arguments to the model
            del step_space['recalibrate_iterations']

            log.info("Step {}".format(index))
            # grid search for best params and update the defaults
            self.params.update(
                self.__do_grid_search(X, y, optimal_br, step_space)
            )
            index += 1

        self.params['learning_rate'] = final_lr

        return self.params

    def __do_grid_search(self, X: DataFrame, y: Series, optimal_boosting_rounds: int, param_grid: dict,
                         log_level=1) -> dict:
        """
        Trains cross-validated model for each combination of the provided hyperparameters, and picks the best based on MAE
        :param X:
        :param y:
        :param param_grid:
        :return:
        """
        # Generate all possible combinations of hyperparameters
        param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]

        best_params = None
        results = []

        if self.direction == OptimizationDirection.MINIMIZE:
            best_score = float('inf')
        elif self.direction == OptimizationDirection.MAXIMIZE:
            best_score = 0
        else:
            log.error("Optimization direction not valid")
            return {}

        for params in param_combinations:

            full_params = self.params.copy()
            full_params.update(params)

            accuracy, _, _ = self.trainer.validate_model(X, y, log_level=0, iterations=optimal_boosting_rounds,
                                                      params=full_params)
            results.append((params, accuracy))

            if (self.direction == OptimizationDirection.MINIMIZE and (accuracy < best_score)) or \
                    (self.direction == OptimizationDirection.MAXIMIZE and (accuracy > best_score)):
                best_score = accuracy
                best_params = params

        if log_level > 0:
            log.result("Best parameters found", best_params)
            log.result("Best accuracy", best_score)

        if log_level > 1:
            # Print all results
            for params, accuracy in results:
                log.detail(f"Parameters: {params}, MAE: {accuracy}")

        return best_params
