import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
from hyperopt import hp
from pytorch_tabnet.tab_model import TabNetClassifier
from src.enums.objective import Objective
from src.models.model_wrapper import ModelWrapper
from src.models.overrides.tabnet_classifier_override import TabNetClassifierOverride


class TabNetClassifierWrapper(ModelWrapper):

    def __init__(self, early_stopping_rounds=20):
        torch.manual_seed(0)
        super().__init__(early_stopping_rounds=early_stopping_rounds)

    def get_objective(self) -> Objective:
        return Objective.CLASSIFICATION

    def get_base_model(self, iterations, params):
        # Tabnet is not compatible with pandas datasets, we'll need an override class.
        params = params.copy()
        # decouple n_d and n_a for direct construction
        if 'n_d_n_a' in params:
            params['n_d'] = params['n_a'] = params.pop('n_d_n_a')
        return TabNetClassifierOverride(**params)

    def get_starter_params(self) -> dict:
        return {
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "n_shared": 2,
            "cat_emb_dim": 1,
            "optimizer_params": {"lr": 2e-2},
            "mask_type": "entmax",
            "optimizer_fn": torch.optim.Adam,
            "lambda_sparse": 1e-3,
            # "cat_idxs": cat_idxs or [],
            # "cat_dims": cat_dims or [],
            "verbose": 0,
        }

    def get_grid_space(self) -> list[dict]:
        return [
            {
                'recalibrate_iterations': False,
                # According to the paper n_d=n_a is usually a good choice, let's group them up
                'n_d_n_a': [8, 12, 16],
                'n_steps': [3, 4, 5],
                'n_shared': [2, 3, 4, 5],
            },
            {
                'recalibrate_iterations': True,
                'cat_emb_dim': [1, 2, 3, 4, 5],
            },
            {
                'recalibrate_iterations': True,
                'mask_type': ["entmax", "sparsemax"],
                'lambda_sparse': np.logspace(-3, -1, num=5)
            }
        ]

    def get_bayesian_space(self) -> dict:
        return {
            # According to the paper n_d=n_a is usually a good choice, let's group them up
            'n_d_n_a': hp.quniform('n_d_n_a', 8, 16, 4),
            'n_steps': hp.quniform('n_steps', 3, 5, 1),
            'n_shared': hp.quniform('n_shared', 2, 5, 1),
            'cat_emb_dim': hp.quniform('cat_emb_dim', 1, 5, 1),
            # 'lr': hp.uniform('lr', 2e-4, 2e-2),
            'mask_type': hp.choice('mask_type', ["entmax", "sparsemax"]),
            'lambda_sparse': hp.loguniform('lambda_sparse', 1e-3, 3e-3),
        }

    def fit(self, X, y, iterations, params=None):
        params = params or {}
        params = params.copy()

        # decouple n_d and n_a
        if 'n_d_n_a' in params:
            params['n_d'] = params['n_a'] = params['n_d_n_a']
            del params['n_d_n_a']

        self.model: TabNetClassifier = TabNetClassifier(
            **params
        )

        self.model.fit(
            X_train=X.to_numpy(),
            y_train=y.to_numpy(),
            eval_metric=["accuracy"],
            max_epochs=iterations + 1,
            patience=self.early_stopping_rounds,
            batch_size=1024,
            virtual_batch_size=128,
            drop_last=False
        )

    def train_until_optimal(self, train_X, validation_X, train_y, validation_y, params=None):
        params = params or {}
        params = params.copy()

        # decouple n_d and n_a
        if 'n_d_n_a' in params:
            params['n_d'] = params['n_a'] = params['n_d_n_a']
            del params['n_d_n_a']

        self.model: TabNetClassifier = TabNetClassifier(
            **params
        )
        self.model.fit(
            X_train=train_X.to_numpy(),
            y_train=train_y.to_numpy(),
            eval_set=[(validation_X.to_numpy(), validation_y.to_numpy())],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=2000,
            patience=self.early_stopping_rounds,
            batch_size=1024,
            virtual_batch_size=128,
            drop_last=False
        )

    def predict(self, X) -> any:
        return pd.Series(self.model.predict(X.to_numpy()))

    def predict_proba(self, X) -> any:
        return pd.Series(self.model.predict_proba(X.to_numpy())[:, 1])

    def get_best_iteration(self) -> int:
        # get callbacks container, blatantly ignoring private accessor
        callbacks = self.model._callback_container.callbacks
        # if there is more than one callback, chance is that the second is the early stopping callback
        if len(callbacks) > 1:
            # try to get the best epoch, or give up and return 0
            return callbacks[1].best_epoch or 0
        else:
            return 0

    def get_loss(self) -> dict[str, dict[str, list[float]]]:
        if self.model is None:
            raise ValueError("No model has been fitted")

        history = self.model.history.history
        losses = [value for key, value in history.items() if 'valid_' in key]

        return {'0': {'loss': next(iter(losses))}}

    def get_feature_importance(self, features) -> DataFrame:
        if self.model is None:
            raise ValueError("No model has been fitted")

        importances = self.model.feature_importances_

        # sort and merge importances and column names into a dataframe
        feature_importances = sorted(zip(importances, features), reverse=True)
        sorted_importances, sorted_features = zip(*feature_importances)
        return pd.DataFrame({'feats': sorted_features[:50], 'importance': sorted_importances[:50]})
