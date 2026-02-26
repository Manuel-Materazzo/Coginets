import time

from src.enums.accuracy_metric import AccuracyMetric
from src.enums.optimization_direction import OptimizationDirection
from src.models.xgb_regressor import XGBRegressorWrapper
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.preprocessors.empty_data_preprocessor import EmptyDataPreprocessor
from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
from src.hyperparameter_optimizers.default_grid_optimizer import DefaultGridOptimizer
from src.hyperparameter_optimizers.hyperopt_bayesian_optimizer import HyperoptBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer
from src.trainers.trainer import save_model
from src.utils.data_utils import load_data


print("Loading data...")
X, y = load_data('train.csv', 'SalePrice')

# save model file for current dataset on target directory
print("Saving data model...")
save_data_model(X)

# instantiate data pipeline and preprocessor
preprocessor = EmptyDataPreprocessor()
pipeline = HousingPricesCompetitionDTPipeline(X)

# preprocess data
preprocessor.preprocess_data(X)

# pick a model, a trainer and an optimizer
model_type = XGBRegressorWrapper(early_stopping_rounds=10)
trainer = CachedAccurateCrossTrainer(pipeline, model_type, X, y, metric=AccuracyMetric.MAE, grouping_columns=None)
optimizer = DefaultGridOptimizer(trainer, model_type, direction=OptimizationDirection.MINIMIZE)

# optimize parameters
print("Tuning Hyperparameters...")
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03)
end = time.time()

print("Tuning took {} seconds".format(end - start))

print("Training and evaluating model...")
_, boost_rounds, _ = trainer.validate_model(X, y, log_level=1, params=optimized_params)
print()

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=boost_rounds, params=optimized_params)

# save preprocessor on target directory
print("Saving preprocessor...")
preprocessor.save_preprocessor()

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)
