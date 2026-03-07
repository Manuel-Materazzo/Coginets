import os
import unittest
import pytest

from src.pipelines.dt_pipeline import load_pipeline, save_data_model
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from tests.data_load import load_regression_data


class TestPipelinesUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_regression_data()

    def test_pipeline_find_ohe(self):
        pipeline = HousingPricesCompetitionDTPipeline(self.X)
        ohe = pipeline.find_one_hot_encoder(pipeline.pipeline)
        self.assertIsNone(ohe)

    def test_pipeline_save(self):
        pipeline = HousingPricesCompetitionDTPipeline(self.X)
        pipeline.save_pipeline()
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../target/pipeline.pkl'
        self.assertTrue(os.path.exists(file_path))

    def test_data_model_save(self):
        save_data_model(self.X)
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../target/data-model.json'
        self.assertTrue(os.path.exists(file_path))

    @pytest.mark.skipif("CI" in os.environ)
    def test_pipeline_load(self):
        pipeline = load_pipeline()
        self.assertIsNotNone(pipeline)
