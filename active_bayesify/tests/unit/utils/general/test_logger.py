import unittest
import os
from pyfakefs.fake_filesystem_unittest import TestCase
from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.utils.general.logger import Logger


class LoggerTest(TestCase):
    logger = None
    file_name = "test"

    def setUp(self):
        self.setUpPyfakefs()
        self.fs.create_file("config.ini", contents=f"[Paths]\nLogs = ./\n[Logging]\nFileName = {self.file_name}\nLogLevel = INFO")
        self.config_parser = ConfigParser("x264")
        self.logger = Logger(self.config_parser).get_logger()

    def test_log_warning(self):
        self.logger.warning("test")
        log_file = open(self.get_log_file(), "r")
        content = log_file.read()
        self.assertIn("WARNING: test", content)

    def test_log_info(self):
        self.logger.info("test")
        log_file = open(self.get_log_file(), "r")
        content = log_file.read()
        self.assertIn("INFO: test", content)

    def get_log_file(self) -> str:
        """
        Scans directory to find log file.

        :return: string file name of log file.
        """
        with os.scandir(self.config_parser.get("Paths", "Logs")) as directory:
            for file in directory:
                if self.file_name in file.name:
                    return file.name


if __name__ == '__main__':
    unittest.main()
