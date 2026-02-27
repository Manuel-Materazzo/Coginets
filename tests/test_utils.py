import unittest

import numpy as np
import pandas as pd

from src.utils.json_utils import map_dtype
from src.utils.data_utils import load_data
from src.utils.time_transformation_utils import generate_daily_features


class TestJsonUtils(unittest.TestCase):

    def test_map_integer_dtype(self):
        self.assertEqual(map_dtype(pd.Series([1, 2, 3]).dtype), 'int')

    def test_map_float_dtype(self):
        self.assertEqual(map_dtype(pd.Series([1.0, 2.0]).dtype), 'float')

    def test_map_string_dtype(self):
        self.assertEqual(map_dtype(pd.Series(['a', 'b']).dtype), 'str')

    def test_map_unknown_dtype(self):
        self.assertEqual(map_dtype(pd.Series([True, False]).dtype), 'str')


class TestDataUtils(unittest.TestCase):

    def test_load_regression_data(self):
        X, y = load_data('regression_unit_test.csv', 'SalePrice')
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertNotIn('SalePrice', X.columns)
        self.assertEqual(len(X), len(y))

    def test_load_classification_data(self):
        X, y = load_data('classification_unit_test.csv', 'SalePrice')
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)


class TestTimeTransformationUtils(unittest.TestCase):

    def test_generate_daily_features(self):
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 10:30:00', '2024-01-01 14:45:00'])
        })
        result = generate_daily_features(df, 'timestamp')
        self.assertIn('hour', result.columns)
        self.assertIn('minute', result.columns)
        self.assertIn('hour_sin', result.columns)
        self.assertIn('hour_cos', result.columns)
        self.assertIn('minute_sin', result.columns)
        self.assertIn('minute_cos', result.columns)

    def test_generate_daily_features_with_seconds(self):
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 10:30:15'])
        })
        result = generate_daily_features(df, 'timestamp', seconds=True)
        self.assertIn('seconds', result.columns)
        self.assertIn('seconds_sin', result.columns)
        self.assertIn('seconds_cos', result.columns)
