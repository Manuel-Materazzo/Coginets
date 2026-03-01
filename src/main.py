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
from src.utils.logger import log

log.header("Single Model Training")

with log.group("Setup"):
    log.info("Loading data...")
    X, y = load_data('train.csv', 'SalePrice')
    log.success("Data loaded")

    log.info("Saving data model...")
    save_data_model(X)
    log.success("Data model saved")

# instantiate data pipeline and preprocessor
preprocessor = EmptyDataPreprocessor()
pipeline = HousingPricesCompetitionDTPipeline(X)

# preprocess data
preprocessor.preprocess_data(X)

# pick a model, a trainer and an optimizer
model_type = XGBRegressorWrapper(early_stopping_rounds=10)
trainer = CachedAccurateCrossTrainer(pipeline, model_type, X, y, metric=AccuracyMetric.MAE, grouping_columns=None)
optimizer = DefaultGridOptimizer(trainer, model_type, direction=OptimizationDirection.MINIMIZE)

with log.group("Hyperparameter Tuning"):
    start = time.time()
    optimized_params = optimizer.tune(X, y, 0.03)
    elapsed = time.time() - start
    log.result("Time elapsed", "{:.1f}s".format(elapsed))

with log.group("Model Evaluation"):
    _, boost_rounds, _ = trainer.validate_model(X, y, log_level=1, params=optimized_params)

with log.group("Full Training"):
    complete_model, _ = trainer.train_model(X, y, iterations=boost_rounds, params=optimized_params)
    log.success("Model trained")

with log.group("Saving Artifacts"):
    preprocessor.save_preprocessor()
    log.success("Preprocessor saved")

    pipeline.save_pipeline()
    log.success("Pipeline saved")

    save_model(complete_model)
    log.success("Model saved")
