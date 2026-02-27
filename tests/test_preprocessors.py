import os
import unittest

import pandas as pd

from src.preprocessors.data_preprocessor import DataPreprocessor, load_preprocessor
from src.preprocessors.empty_data_preprocessor import EmptyDataPreprocessor
from tests.data_load import load_regression_data


class TestEmptyDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()

    def test_preprocess_data_no_change(self):
        original = self.X.copy()
        preprocessor = EmptyDataPreprocessor()
        preprocessor.preprocess_data(self.X)
        pd.testing.assert_frame_equal(self.X, original)

    def test_is_data_preprocessor(self):
        preprocessor = EmptyDataPreprocessor()
        self.assertIsInstance(preprocessor, DataPreprocessor)

    def test_save_preprocessor(self):
        preprocessor = EmptyDataPreprocessor()
        preprocessor.save_preprocessor()
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../target/preprocessor.pkl'
        self.assertTrue(os.path.exists(file_path))

    def test_load_preprocessor(self):
        preprocessor = EmptyDataPreprocessor()
        preprocessor.save_preprocessor()
        loaded = load_preprocessor()
        self.assertIsInstance(loaded, EmptyDataPreprocessor)
