import optuna
import os
import platform
import optuna_distributed
from pandas import DataFrame, Series

from src.enums.optimization_direction import OptimizationDirection
from src.hyperparameter_optimizers.hp_optimizer import HyperparameterOptimizer
from src.models.model_wrapper import ModelWrapper
from src.trainers.trainer import Trainer


class OptunaOptimizer(HyperparameterOptimizer):
    def __init__(self, trainer: Trainer, model_wrapper: ModelWrapper,
                 direction: OptimizationDirection = OptimizationDirection.MINIMIZE):
        super().__init__(trainer, model_wrapper, direction=direction)
        self.y = None
        self.X = None
        self.study = None
        self.trials = 100
        self.domain_space = model_wrapper.get_bayesian_space()

    def show_param_importance(self):
        optuna.visualization.plot_param_importances(self.study)

    def tune(self, X: DataFrame, y: Series, final_lr: float) -> dict:
        """
        Calculates the best hyperparameters for the dataset by performing a bayesian optimization
        Trains cross-validated model for each combination of hyperparameters, and picks the best based on MAE.
        :param X:
        :param y:
        :param final_lr:
        :return:
        """
        self.X = X
        self.y = y

        self.study = optuna.create_study(direction=self.direction.value.lower())
        # leverage distributed training on linux, but not in CI environments
        # where subprocess forking causes joblib/loky workers to deadlock
        in_ci = os.environ.get('CI', 'false').lower() == 'true'
        if platform.system() != 'Windows' and not in_ci:
            self.study = optuna_distributed.from_study(self.study)
        self.study.optimize(self.__objective, n_trials=self.trials)
        self.params.update(self.study.best_params)

        self.params['learning_rate'] = final_lr

        return self.params

    def __objective(self, trial):
        """
        Defines the objective function to be minimized.
        Trains a model with hyperparameters and returns the cross validated MAE.
        :return:
        """

        octuna_space = {}

        # convert hyperopt space to optuna by doing a great deal of dark magic
        for key in self.domain_space:
            # extract arguments of the hpspace
            args = self.domain_space[key].pos_args
            space_accessor = args[0].arg
            param_name = space_accessor['label'].obj
            param_type = space_accessor['obj'].name

            # instantiate the correct optuna space
            # NB: key could be different from param_name. This allows for parameter pairing (parameterA = parameterB)
            match param_type:
                case 'uniform':
                    param_low_arg = space_accessor['obj'].arg['low'].obj
                    param_high_arg = space_accessor['obj'].arg['high'].obj
                    octuna_space[key] = trial.suggest_float(param_name, param_low_arg, param_high_arg)
                case 'loguniform':
                    param_low_arg = space_accessor['obj'].arg['low'].obj
                    param_high_arg = space_accessor['obj'].arg['high'].obj
                    octuna_space[key] = trial.suggest_float(param_name, param_low_arg, param_high_arg, log=True)
                case 'quniform':
                    param_low_arg = space_accessor['obj'].arg['low'].obj
                    param_high_arg = space_accessor['obj'].arg['high'].obj
                    param_q = space_accessor['obj'].arg['q'].obj
                    octuna_space[key] = trial.suggest_int(param_name, param_low_arg, param_high_arg, step=param_q)
                case 'randint':
                    options = [val.obj for val in args[1:]]  # extract options from extra args
                    octuna_space[key] = trial.suggest_categorical(param_name, list(options))

        accuracy, _, _ = self.trainer.validate_model(self.X, self.y, log_level=0, params=octuna_space)
        return accuracy