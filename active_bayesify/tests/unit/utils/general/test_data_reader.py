import unittest
from pyfakefs.fake_filesystem_unittest import TestCase
from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.utils.general.data_reader import DataReader


class DataReaderTest(TestCase):

    def setUp(self):
        # mock config ini for config parser
        self.setUpPyfakefs()
        self.fs.create_file("config.ini", contents="[Paths]\nData = ./\nResults = ./")

        # add dependencies
        self.config_parser = ConfigParser("x264")
        self.data_reader = DataReader(self.config_parser)

    def test_get_all_functions(self):
        self.fs.create_file("function_names.csv", contents=",method\n0,test\n")
        function_names = self.data_reader.get_all_functions()
        self.assertEqual(function_names[0], "test")

    def test_get_functions_with_data(self):
        self.fs.create_file("test_0.csv")  # note the repetition info
        function_names = self.data_reader.get_functions_with_data()
        self.assertEqual(function_names[0], "test")

    def test_read_in_data_from_function(self):
        self.fs.create_file("test.csv", contents="index,taskID,method,time,energy,threads\n0,8,1,test,"
                                                 "3.353704929351806,5.914208961058112,2")
        data = self.data_reader.read_in_data("test")
        self.assertEqual(data.loc[0, "method"], "test")
        self.assertEqual(data.loc[0, "threads"], 2)

    def test_read_in_model_results(self):
        self.fs.create_file("test.csv", contents="repetition,iteration,feature,selected_by_lasso,selected_by_ridge,selected_by_p4,lasso_influence,ridge_influence,p4_min_influence,p4_max_influence,p4_influence,lasso_mape,ridge_mape,p4_mape,p4_elpd_psis\n"
                                                 "0,10,\"('h',)\",True,True,True,31.99288853817998,7.649233025145661,19.648441,29.232668,24.440556,0.9590621022660734,0.361167929170101,0.9568779291714152,0.022420310974121")
        data = self.data_reader.read_in_results("test")
        self.assertEqual(data.loc[0, "repetition"], 0)
        self.assertEqual(data.loc[0, "p4_mape"], 0.9568779291714152)


if __name__ == '__main__':
    unittest.main()
