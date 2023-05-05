import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from active_bayesify.tests.unit.helper import get_test_data_pool
from pyfakefs.fake_filesystem_unittest import TestCase
from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.mvp.pipeline_mvp import MvpPipeline
from bayesify.pairwise import PyroMCMCRegressor, P4Preprocessing

option_names = ["threads", "ref", "rc_lookahead", "no_8x8dct",
                   "no_cabac", "no_deblock", "no_mbtree", "no_fast_pskip", "no_mixed_refs"]
option_names_with_id = ["taskID"] + option_names


def get_config_file() -> str:
    """
    Provides full `config.ini` file for tests.

    :return: string to mock config file with.
    """
    return "[Paths]\nData = ./\nResults = ./\nLogs = ./\nImages = ./\nMapeImages = " \
           "./\n\n[Pipeline]\nRuntimeThreshold = 0.05\nBatchSize = 5\nNumberRepetitions = " \
           "5\n\n[Logging]\nLogLevel = INFO\nFileName = mvp_pipeline"


class PipelineMvpTest(TestCase):
    config_parser = None

    def setUp(self):
        self.setUpPyfakefs()
        # mock full `config.ini`
        self.fs.create_file("config.ini", contents=get_config_file())
        self.config_parser = ConfigParser("x264")
        self.mvp_pipeline = MvpPipeline()
        self.test_data_frame = get_test_data_pool()
        self.pause()


    def test_make_prediction(self):
        X_test_numpy = self.test_data_frame[option_names].to_numpy()
        y_test_numpy = self.test_data_frame["energy"].to_numpy()
        model = PyroMCMCRegressor()
        model.fit(X_test_numpy, y_test_numpy, mcmc_samples=100, mcmc_tune=200, feature_names=option_names, mcmc_cores=1)

        prediction = self.mvp_pipeline.make_prediction(model, X_test_numpy, 500)

        self.assertEqual(prediction.shape, (500, 20))

    def test_evaluate_model(self):
        X_test_numpy = self.test_data_frame[option_names].to_numpy()
        y_test_numpy = self.test_data_frame["energy"].to_numpy()
        model = PyroMCMCRegressor()
        model.fit(X_test_numpy, y_test_numpy, mcmc_samples=100, mcmc_tune=200, feature_names=option_names, mcmc_cores=1)

        preprocessor = Mock()
        preprocessor.transform = Mock()
        preprocessor.transform.return_value = X_test_numpy

        prediction_score, _ = self.mvp_pipeline.evaluate_model(model, preprocessor, self.test_data_frame, self.test_data_frame)

        self.assertGreater(prediction_score, 0)

    def test_preprocess_data(self):
        X_test_numpy = self.test_data_frame[option_names].to_numpy()
        y_test_numpy = self.test_data_frame["energy"].to_numpy()

        preprocessor, new_features, new_feature_names = self.mvp_pipeline.preprocess_data(X_test_numpy, y_test_numpy, option_names)

        self.assertIsNotNone(preprocessor)
        self.assertEqual(new_feature_names, [('ref',), ('threads', 'ref'), ('threads', 'rc_lookahead'), ('ref', 'rc_lookahead'), ('rc_lookahead', 'no_cabac'), ('rc_lookahead', 'no_deblock')])
        self.assertIsNone(np.testing.assert_array_equal(new_features[0], [4, 8, 20, 40, 0, 10]))

    def test_train_model(self):
        X_test_numpy = self.test_data_frame[option_names].to_numpy()
        y_test_numpy = self.test_data_frame["energy"].to_numpy()
        preprocessor, new_features, new_feature_names = self.mvp_pipeline.preprocess_data(X_test_numpy, y_test_numpy, option_names)

        model, new_feature_names = self.mvp_pipeline.train_model(new_features, y_test_numpy, new_feature_names)

        self.assertIsNotNone(model)
        self.assertEqual(new_feature_names,
                         [('ref',), ('threads', 'ref'), ('threads', 'rc_lookahead'), ('ref', 'rc_lookahead'),
                          ('rc_lookahead', 'no_cabac'), ('rc_lookahead', 'no_deblock')])

    def test_train_and_evaluate_model(self):
        self.mvp_pipeline.logger = Mock()
        self.mvp_pipeline.logger.info = Mock()
        self.mvp_pipeline.logger.warning = Mock()

        _, prediction_score = self.mvp_pipeline.train_and_evaluate_model("test", 0, 0, self.test_data_frame, self.test_data_frame, self.test_data_frame, self.test_data_frame)

        self.assertEqual(self.mvp_pipeline.logger.info.call_args_list[0].args[0], "test:0:0: model was trained successfully!")
        self.assertGreater(prediction_score, 0)

    def test_fail_on_train_and_evaluate_model(self):
        self.mvp_pipeline.logger = Mock()
        self.mvp_pipeline.logger.info = Mock()
        self.mvp_pipeline.logger.warning = Mock()

        _, prediction_score = self.mvp_pipeline.train_and_evaluate_model("test", 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        self.assertIn("test:0:0: error was thrown", self.mvp_pipeline.logger.warning.call_args_list[0].args[0])
        self.assertIsNone(prediction_score)


if __name__ == '__main__':
    unittest.main()
