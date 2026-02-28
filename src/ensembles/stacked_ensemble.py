import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.linear_model import Ridge

from src.ensembles.ensemble import EnsembleMember, Ensemble
from src.utils.logger import log


class StackedEnsemble(Ensemble):

    def __init__(self, members: list[EnsembleMember]):
        super().__init__(members)
        self.meta_learner = None
        self.trials = None

    def post_validation_callback(self, X: DataFrame, y: Series, oof_predictions: DataFrame):
        """
        Callback to train meta learner when done training. Do not call manually.
        :param oof_predictions:
        :param X:
        :param y:
        :return:
        """
        log.section("OOF Comparisons")
        log.table(oof_predictions)
        log.end_section()
        self._train_meta_learner(oof_predictions, y)

    def show_weights(self):
        """
        Shows ensemble weights
        :return:
        """

        if self.meta_learner is None:
            log.warning("Meta learner not trained. This probably means the validation was skipped or the callback is set incorrectly.")
            return

        # X is metafearures dataframe
        feature_names = self.oof_predictions.columns
        feature_importances = self.meta_learner.coef_

        importance_df = DataFrame({'feature': feature_names, 'importance': feature_importances}).sort_values(
            by='importance', ascending=False)

        log.table(importance_df)
        # plot it!
        plt.figure(figsize=(12, 8))
        plt.xlabel('Coefficient')
        plt.ylabel('Model')
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.show()

    def _train_meta_learner(self, oof_predictions: DataFrame, y: Series):
        """
        Trains a meta-learner on the ensemble models predictions (meta-features).
        :param oof_predictions:
        :param y:
        :return:
        """
        self.meta_learner = Ridge(alpha=0.1, random_state=0, max_iter=self.trials)
        self.meta_learner.fit(oof_predictions, y)

    def predict(self, X: DataFrame) -> Series:
        if len(self.models) == 0:
            log.error("No models trained, use train() first")
            return Series([])

        oof_predictions = DataFrame()

        i = 0
        # for each trained model
        for model in self.models:
            # make prediction and add to the meta-features dataframe
            oof_predictions[model.__class__.__name__ + '_' + str(i)] = model.predict(X)
            i += 1

        # make meta-learner prediction on models output
        weighted_pred = self.meta_learner.predict(oof_predictions)

        return weighted_pred
