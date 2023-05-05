import unittest
import pandas as pd
import numpy as np
from pyfakefs.fake_filesystem_unittest import TestCase
from active_bayesify.final.evaluation import ActiveBayesifyEvaluation
from active_bayesify.utils.general.config_parser import ConfigParser
from pandas.testing import assert_frame_equal


def get_config_file() -> str:
    """
    Provides full `config.ini` file for tests.

    :return: string to mock config file with.
    """
    return "[Paths]\nData = ./\nResults = ./\nLogs = ./\nImages = ./\nMapeImages = " \
           "./\n\n[Pipeline]\nRuntimeThreshold = 0.05\nBatchSize = 5\nNumberRepetitions = " \
           "5\n\n[Logging]\nLogLevel = INFO\nFileName = mvp_pipeline"


class TestEvaluation(TestCase):
    config_parser = None

    def setUp(self):
        self.setUpPyfakefs()
        # mock full `config.ini`
        self.fs.create_file("config.ini", contents=get_config_file())
        self.config_parser = ConfigParser("x264")
        self.evaluation = ActiveBayesifyEvaluation("x264")
        self.pause()

    def test_increment_repetition_and_iteration_test_normal_step(self):
        test = pd.DataFrame(columns=["repetition", "iteration"], data=[[0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10]])  # add assertion here

        rep, ite = self.evaluation.increment_repetition_and_iteration(test, 0, 2)

        self.assertEqual(rep, 0)
        self.assertEqual(ite, 4)

    def test_increment_repetition_and_iteration_test_rep_leap(self):
        test = pd.DataFrame(columns=["repetition", "iteration"], data=[[0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10]])  # add assertion here

        rep, ite = self.evaluation.increment_repetition_and_iteration(test, 0, 10)

        self.assertEqual(rep, 1)
        self.assertEqual(ite, 2)

    def test_increment_repetition_and_iteration_test_max(self):
        test = pd.DataFrame(columns=["repetition", "iteration"], data=[[0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10]])  # add assertion here

        rep, ite = self.evaluation.increment_repetition_and_iteration(test, 1, 10)

        self.assertIsNone(rep)
        self.assertIsNone(ite)


    def test_calculate_ci_interval_width_change(self):
        columns = ['is_active', 'model_name', 'repetition', 'iteration', 'function_name', 'feature', 'influence', 'min_influence', 'max_influence', 'mape']
        data = [
            [True, 'p4', 0, 10, 'get_ref', "('ref',)", 10, 8, 12, 1],
            [True, 'p4', 0, 15, 'get_ref', "('ref',)", 15, 14, 16, 1]
        ]
        test = pd.DataFrame(columns=columns, data=data)

        expected_columns = ["is_active", "model_name", "repetition", "iteration", "function_name", "feature", "ci_interval_width", "ci_interval_width_change"]
        expected_data = [
            [True, "p4", 0, 10, "get_ref", "('ref',)", 4, -2],
            [True, "p4", 0, 15, "get_ref", "('ref',)", 2, np.nan]
        ]
        expected = pd.DataFrame(columns=expected_columns, data=expected_data)

        result = self.evaluation.calculate_ci_interval_width_change(test, ["p4"])

        assert_frame_equal(result, expected, check_dtype=False)





if __name__ == '__main__':
    unittest.main()
