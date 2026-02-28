import time

from src.ensembles.weighted_ensemble import WeightedEnsemble
from src.models.xgb_regressor import XGBRegressorWrapper
from src.models.lgbm_regressor import LGBMRegressorWrapper
from src.models.catboost_regressor import CatBoostRegressorWrapper

from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline

from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer

from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer

from src.trainers.trainer import save_model
from src.utils.data_utils import load_data
from src.utils.logger import log

log.header("Ensemble Training")

with log.group("Setup"):
    log.info("Loading data...")
    X, y = load_data('train.csv', 'SalePrice')
    log.success("Data loaded")

    log.info("Saving data model...")
    save_data_model(X)
    log.success("Data model saved")

# instantiate data pipeline
pipeline = HousingPricesCompetitionDTPipeline(X)

# create model trainer and optimizer for catboost
catboost_model_type = CatBoostRegressorWrapper()
catboost_trainer = CachedAccurateCrossTrainer(pipeline, catboost_model_type, X, y)
catboost_optimizer = OptunaOptimizer(catboost_trainer, catboost_model_type)

# define an ensemble of an XGBoost model with predefined params, and a CatBoost model with auto-optimization
ensemble = WeightedEnsemble(members=[
    {
        'trainer': CachedAccurateCrossTrainer(pipeline, XGBRegressorWrapper(), X, y),
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'n_jobs': -1
        },
        'optimizer': None
    },
    {
        'trainer': catboost_trainer,
        'params': None,
        'optimizer': catboost_optimizer
    }
])

with log.group("Model Validation & Hyperparameter Tuning"):
    start = time.time()
    ensemble.validate_models_and_show_leaderboard(X, y)
    elapsed = time.time() - start
    log.result("Time elapsed", "{:.1f}s".format(elapsed))

with log.group("Ensemble Weights"):
    ensemble.show_weights()

with log.group("Full Ensemble Training"):
    ensemble.train(X, y)
    log.success("Ensemble trained")

with log.group("Saving Artifacts"):
    pipeline.save_pipeline()
    log.success("Pipeline saved")

    save_model(ensemble)
    log.success("Model saved")
