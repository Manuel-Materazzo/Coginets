from pytorch_tabnet.tab_model import TabNetClassifier


class TabNetClassifierOverride(TabNetClassifier):
    def fit(self, X_train, y_train, eval_set=None, eval_name=None, eval_metric=None, loss_fn=None, weights=0,
            max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128, num_workers=0, drop_last=True,
            callbacks=None, pin_memory=True, from_unsupervised=None, warm_start=False, augmentations=None,
            compute_importance=True):
        """
        Overrides the default fit method of TabNetRegressor, converts Pandas DataFrame to numpy array before processing.
        :param X_train:
        :param y_train:
        :param eval_set:
        :param eval_name:
        :param eval_metric:
        :param loss_fn:
        :param weights:
        :param max_epochs:
        :param patience:
        :param batch_size:
        :param virtual_batch_size:
        :param num_workers:
        :param drop_last:
        :param callbacks:
        :param pin_memory:
        :param from_unsupervised:
        :param warm_start:
        :param augmentations:
        :param compute_importance:
        :return:
        """
        # Convert pandas DataFrame to numpy arrays
        numpy_X_train = X_train.to_numpy()
        numpy_y_train = y_train.to_numpy()

        if eval_set is not None:
            eval_set = [(X.to_numpy(), y.to_numpy()) for X, y in eval_set]

        # Call the original fit method with numpy arrays
        super().fit(numpy_X_train, numpy_y_train, eval_set=eval_set, eval_name=eval_name, eval_metric=eval_metric,
                    loss_fn=loss_fn, weights=weights, max_epochs=max_epochs, patience=patience,
                    batch_size=batch_size, virtual_batch_size=virtual_batch_size, num_workers=num_workers,
                    drop_last=drop_last, callbacks=callbacks, pin_memory=pin_memory,
                    from_unsupervised=from_unsupervised, warm_start=warm_start, augmentations=augmentations,
                    compute_importance=compute_importance)