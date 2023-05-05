import unittest
from pyfakefs.fake_filesystem_unittest import TestCase
from active_bayesify.utils.general.config_parser import ConfigParser


class ConfigParserTest(TestCase):
    config_parser = None

    def setUp(self):
        self.setUpPyfakefs()
        self.fs.create_file("config.ini", contents="[StringTest]\nTest = string\n\n"
                                                   "[PathTest]\nResults = ./results/\nData = ../data/\n\n"
                                                   "[IntegerTest]\nTest = 1234")
        self.config_parser = ConfigParser("x264")

    def test_read_file(self):
        self.assertIsNotNone(self.config_parser.sections())

    def test_read_string_option(self):
        test_string = self.config_parser.get("StringTest", "Test")
        self.assertEqual(test_string, "string")

    def test_read_integer_option(self):
        test_integer = self.config_parser.getint("IntegerTest", "Test")
        self.assertEqual(test_integer, 1234)

    def test_read_string_option_with_system_name(self):
        test_string = self.config_parser.get_path_with_system_name("Results")
        self.assertEqual(test_string, "./results/x264/")

    def test_read_string_option_with_system_name(self):
        test_string = self.config_parser.get_path_with_system_name("Data")
        self.assertEqual(test_string, "../data/x264/")


if __name__ == '__main__':
    unittest.main()
