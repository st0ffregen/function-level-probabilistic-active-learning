import unittest

from active_bayesify.final.pipeline import ActiveBayesify
from active_bayesify.tests.unit.helper import get_test_data_pool
from pyfakefs.fake_filesystem_unittest import TestCase
from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.mvp.pipeline_mvp import MvpPipeline

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
           "5\n\n[Logging]\nLogLevel = INFO\nFileName = pipeline"


class PipelineMvpTest(TestCase):
    config_parser = None

    def setUp(self):
        self.setUpPyfakefs()
        # mock full `config.ini`
        self.fs.create_file("config.ini", contents=get_config_file())
        self.config_parser = ConfigParser(system_name)
        self.pipeline = ActiveBayesify(system_name, 5, )
        self.test_data_frame = get_test_data_pool()
        self.pause()





if __name__ == '__main__':
    unittest.main()
