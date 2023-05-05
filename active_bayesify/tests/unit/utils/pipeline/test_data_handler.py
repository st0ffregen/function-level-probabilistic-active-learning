import copy
import unittest

import pandas as pd
from active_bayesify.tests.unit.helper import get_test_data_pool
from pyfakefs.fake_filesystem_unittest import TestCase
import numpy as np
from active_bayesify.utils.dtos.model_result import ModelResult
from active_bayesify.utils.models.feature import Feature
from active_bayesify.utils.models.p4_influences import P4Influences
from active_bayesify.utils.pipeline.data_handler import DataHandler
from active_bayesify.utils.general.data_reader import DataReader
from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.tests.unit.helper import feature_names_with_id
from pandas.testing import assert_frame_equal

y_columns = ["taskID", "energy", "method"]


class DataHandlerTest(TestCase):
    config_parser = None

    def setUp(self):
        self.setUpPyfakefs()
        self.fs.create_file("config.ini", contents="[Paths]\nData = ./\nResults = ./")

        # add dependencies
        config_parser = ConfigParser("x264")
        data_reader = DataReader(config_parser)
        self.data_handler = DataHandler(data_reader, ["threads", "ref", "rc_lookahead", "no_8x8dct",
                                                      "no_cabac", "no_deblock", "no_mbtree", "no_fast_pskip",
                                                      "no_mixed_refs"])
        self.test_data_pool = get_test_data_pool()
        self.test_x_train = pd.DataFrame(columns=self.test_data_pool.columns, data=self.test_data_pool.sample(n=10))

    def test_split_data(self):
        X_train_init, X_train_al, X_test, y_train_init, y_train_al, y_test = self.data_handler.split_data(
            self.test_data_pool, 10, 7, 0)

        self.assertEqual(X_train_init.shape[0], 10)
        self.assertEqual(X_train_al.shape[0], 7)
        self.assertEqual(X_test.shape[0], 3)
        self.assertEqual(y_train_init.shape[0], 10)
        self.assertEqual(y_train_al.shape[0], 7)
        self.assertEqual(y_test.shape[0], 3)

        self.assertIn(X_train_init.iloc[0, 0], list(range(21)))

    def test_randomly_sample_new_instances(self):
        X_pool = self.test_data_pool[feature_names_with_id]
        y_pool = self.test_data_pool[y_columns]

        X_instances, y_instances = self.data_handler.randomly_sample_new_instances(X_pool, y_pool, 5, 0)

        self.assertEqual(X_instances.shape[0], 5)
        self.assertEqual(y_instances.shape[0], 5)

    def test_actively_sample_new_instances(self):
        X_pool = self.test_data_pool[feature_names_with_id]
        y_pool = self.test_data_pool[y_columns]
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_dict={("threads",): np.array([10, 15]), ("ref",): np.array([2, 4])}),
            P4Influences(function_name="func2", p4_influences_as_dict={("rc_lookahead",): np.array([-8, -5]), ("no_fast_pskip",): np.array([1, 2]), ("no_mixed_refs",): np.array([1, 2])})
        ]
        functions_with_weights = {"func1": 0.6, "func2": 0.4}

        X_instances, y_instances, _, _, _, _ = self.data_handler.actively_sample_new_instances(X_pool, y_pool,
                                                                                               self.test_x_train,
                                                                                               p4_influences_list, 5, 0,
                                                                                               weight_by_vif=True,
                                                                                               functions_with_weights=functions_with_weights)

        self.assertEqual(X_instances.shape[0], 5)
        self.assertEqual(y_instances.shape[0], 5)

    def test_actively_sample_new_instances_with_less_than_batch_size_uncertain_feature_array(self):
        X_pool = self.test_data_pool[feature_names_with_id]
        y_pool = self.test_data_pool[y_columns]
        options_to_be_included = [Feature(option1="threads", influence=np.array([10, 15])), Feature(option1="ref", influence=np.array([2, 4])), Feature(option1="rc_lookahead", influence=np.array([-8, -5]))]
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_feature_list=options_to_be_included)
        ]
        functions_with_weights = {"func1": 0.6}

        # act
        X_instances, y_instances, _, _, _, _ = self.data_handler.actively_sample_new_instances(X_pool, y_pool,
                                                                                               self.test_x_train,
                                                                                               p4_influences_list, 5, 0,
                                                                                               weight_by_vif=True,
                                                                                               functions_with_weights=functions_with_weights)
        # assert
        at_least_one_option_occurs_double = False

        for option in options_to_be_included:
            if X_instances[X_instances[option.option1] > 0].shape[0] >= 2:
                at_least_one_option_occurs_double = True
                break

        self.assertTrue(at_least_one_option_occurs_double)
        self.assertEqual(X_instances.shape[0], 5)
        self.assertEqual(y_instances.shape[0], 5)

    def test_actively_sample_new_instances_with_interaction_feature(self):
        X_pool = self.test_data_pool[feature_names_with_id]
        y_pool = self.test_data_pool[y_columns]
        options_to_be_included = [Feature(option1="no_mixed_refs", option2="no_cabac", influence=np.array([10, 15])), Feature(option1="ref", influence=np.array([2, 4]))]
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_feature_list=options_to_be_included)
        ]
        functions_with_weights = {"func1": 0.6}

        # act
        X_instances, y_instances, _, _, _, _ = self.data_handler.actively_sample_new_instances(X_pool, y_pool,
                                                                                               self.test_x_train,
                                                                                               p4_influences_list, 5, 0,
                                                                                               weight_by_vif=True,
                                                                                               functions_with_weights=functions_with_weights)

        # assert
        self.assertTrue(X_instances[(X_instances[options_to_be_included[0].option1] > 0) & (X_instances[options_to_be_included[0].option2] > 0)].shape[0] > 0)
        self.assertEqual(X_instances.shape[0], 5)
        self.assertEqual(y_instances.shape[0], 5)

    def test_actively_sample_new_instances_with_interaction_feature_returns_instances_without_interaction_because_interaction_never_on(self):
        X_pool = self.test_data_pool[feature_names_with_id].head(5)
        y_pool = self.test_data_pool[y_columns]
        options_to_be_included = [Feature(option1="no_8x8dct", option2="no_mbtree", influence=np.array([10, 15])), Feature(option1="ref", influence=np.array([-8, -5]))]
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_feature_list=options_to_be_included)
        ]
        functions_with_weights = {"func1": 0.6}

        X_instances, y_instances, _, _, _, _ = self.data_handler.actively_sample_new_instances(X_pool, y_pool,
                                                                                               self.test_x_train,
                                                                                               p4_influences_list, 5, 0,
                                                                                               weight_by_vif=True,
                                                                                               functions_with_weights=functions_with_weights)

        self.assertTrue(X_instances[(X_instances[options_to_be_included[0].get_option1()] > 0) & (X_instances[options_to_be_included[0].get_option2()] > 0)].shape[0] == 0)
        self.assertEqual(X_instances.shape[0], 5)
        self.assertEqual(y_instances.shape[0], 5)

    def test_actively_sample_new_instances_every_feature_has_at_least_one_row_where_set_to_on(self):
        """
        This tests checks if the options got correctly sorted so that the options with the least available options
        get sampled first to avoid that options with more samples reduce the selection of possible configs.
        If option "threads" allows 8 configs to be sampled and "ref" only 1 where "threads" is also set "on" and
        "threads" get sampled first and selects this one option than there would be no option left for "ref".
        """
        X_pool = self.test_data_pool[feature_names_with_id]
        y_pool = self.test_data_pool[y_columns]
        options_to_be_included = [Feature(option1="threads", influence=np.array([10, 15])), Feature(option1="ref", influence=np.array([-8, -5])), Feature(option1="rc_lookahead", influence=np.array([-8, -5])), Feature(option1="no_fast_pskip", influence=np.array([-8, -5])), Feature(option1="no_mixed_refs", influence=np.array([-8, -5]))]
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_feature_list=options_to_be_included)
        ]
        functions_with_weights = {"func1": 0.6}

        X_instances, y_instances, _, _, _, _ = self.data_handler.actively_sample_new_instances(X_pool, y_pool,
                                                                                               self.test_x_train,
                                                                                               p4_influences_list, 5, 0,
                                                                                               weight_by_vif=True,
                                                                                               functions_with_weights=functions_with_weights)

        for i in range(len(options_to_be_included)):
            if i < X_instances.shape[0]:
                self.assertTrue(X_instances[options_to_be_included[i].get_option1()].shape[0] > 0)

    def test_drop_used_instances(self):
        columns = ["", "index", "taskID", "method", "time", "energy", "threads", "ref", "rc_lookahead", "no_8x8dct",
                   "no_cabac", "no_deblock", "no_mbtree", "no_fast_pskip", "no_mixed_refs"]
        data = [[8, 8, 1, "test", 3.353704929351806, 5.914208961058112, 2, 4, 10, 0, 0, 1, 0, 1, 1],
                [260, 260, 2, "test", 3.192512989044189, 5.483424730960388, 4, 2, 160, 1, 0, 1, 0, 1, 1]]

        used_instances = pd.DataFrame(data=data, columns=columns)
        thinned_set = self.data_handler.drop_used_instances(self.test_data_pool, used_instances, ["taskID"])

        self.assertEqual(thinned_set.shape[0], 18)

    def test_add_new_trainings_data(self):
        X_orig = self.test_data_pool[feature_names_with_id]
        y_orig = self.test_data_pool[y_columns]

        columns = ["", "index", "taskID", "method", "time", "energy", "threads", "ref", "rc_lookahead", "no_8x8dct",
                   "no_cabac",
                   "no_deblock", "no_mbtree", "no_fast_pskip", "no_mixed_refs"]
        data = [[8, 8, 1, "test", 3.353704929351806, 5.914208961058112, 2, 4, 10, 0, 0, 1, 0, 1, 1],
                [260, 260, 2, "test", 3.192512989044189, 5.483424730960388, 4, 2, 160, 1, 0, 1, 0, 1, 1]]

        new_instances = pd.DataFrame(data=data, columns=columns)
        X_new = new_instances[feature_names_with_id]
        y_new = new_instances[y_columns]

        X_extended, y_extended = self.data_handler.add_new_trainings_data(X_orig, X_new, y_orig, y_new)

        self.assertEqual(X_extended.shape[0], 22)
        self.assertEqual(y_extended.shape[0], 22)

    def test_is_function_filtered_out(self):
        is_function_filtered_out = self.data_handler.is_function_filtered_out(self.test_data_pool, 55, 0.5)

        self.assertFalse(is_function_filtered_out)


    def test_prepare_model_results(self):
        expected = [ModelResult("p4", 0, 0, Feature(option1="test", influence=np.array([1.0, 1.0])), 1.0, 1.0, 1.0, 1.0)]

        model_results = self.data_handler.prepare_model_results(0, 0, ["p4"], [P4Influences(function_name="func", p4_influences_as_feature_list=[Feature(option1="test", influence=np.array([1.0, 1.0]))])], [1.0])

        self.assertTrue(model_results[0] == expected[0])

    def test_write_lasso_p4_data_to_data_frame(self):
        columns = ["model_name", "repetition", "iteration", "function_name", "feature", "influence", "min_influence", "max_influence", "mape"]
        test = pd.DataFrame(columns=columns)
        expected = pd.DataFrame(columns=columns, data=[["p4", 0, 0, "func", "('test',)", 1.0, 1.0, 1.0, 1.0], ["lasso", 0, 0, "func", "('test',)", 1.0, np.nan, np.nan, 1.0]])
        p4_influences_list = [
            P4Influences(function_name="func", p4_influences_as_feature_list=[Feature(option1="test", influence=np.array([1.0, 1.0]))]),
            P4Influences(function_name="func", p4_influences_as_feature_list=[Feature(option1="test", influence=1.0)]),
        ]
        model_results = self.data_handler.prepare_model_results(0, 0, ["p4", "lasso"], p4_influences_list, [1.0, 1.0], "func")

        result = self.data_handler.write_model_results_to_data_frame(test, model_results)

        assert_frame_equal(result, expected, check_dtype=False)

    def test_append_row(self):
        test = pd.DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        new_row = pd.Series({"a": 5, "b": 6})
        expected = pd.DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4], [5, 6]])

        result = self.data_handler.append_row(test, new_row)

        assert_frame_equal(result, expected)

    def test_get_average_p4_influences(self):
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_dict={('no_8x8dct',): np.array([10, 15]), ('no_deblock', 'no_cabac'): np.array([2, 4])}),
            P4Influences(function_name="func2", p4_influences_as_dict={('no_8x8dct',): np.array([-8, -5])})
        ]
        p4_influences_expected = P4Influences(function_name="system", p4_influences_as_dict={('no_8x8dct',): np.array([1, 5]), ('no_deblock', 'no_cabac'): np.array([1, 2])})

        p4_influences_result = self.data_handler.get_average_p4_influences(copy.deepcopy(p4_influences_list), 2)

        self.assertEqual(p4_influences_result, p4_influences_expected)

    def test_get_weighted_p4_influences(self):
        p4_influences_list = [
            P4Influences(function_name="func1", p4_influences_as_dict={('no_8x8dct',): np.array([10, 15]), ('no_deblock', 'no_cabac'): np.array([2, 4])}),
            P4Influences(function_name="func2", p4_influences_as_dict={('no_8x8dct',): np.array([-8, -5])})
        ]
        p4_influences_expected = P4Influences(function_name="system", p4_influences_as_dict={('no_8x8dct',): np.array([2.8, 7]), ('no_deblock', 'no_cabac'): np.array([1.2, 2.4])})

        p4_influences_result = self.data_handler.get_weighted_p4_influences_by_function_runtime(p4_influences_list, {"func1": 0.6, "func2": 0.4})

        np.testing.assert_equal(p4_influences_result, p4_influences_expected)

    def test_get_least_frequent_feature_value_feature_is_single_option(self):
        test = pd.DataFrame(columns=["a", "b"], data=[[1, 2], [2, 3], [3, 3], [3, 3]])

        least_frequent_value = self.data_handler.get_least_frequent_feature_value(test, Feature(option1="b"))

        self.assertEqual(least_frequent_value, 2) # 2 because index[-1] get value from last row

    def test_get_least_frequent_feature_value_feature_is_two_options(self):
        test = pd.DataFrame(columns=["a", "b"], data=[[1, 2], [2, 3], [3, 3], [3, 3]])

        least_frequent_value = self.data_handler.get_least_frequent_feature_value(test, Feature(option1="a", option2="b"))

        self.assertEqual(least_frequent_value, (2, 3))

    def test_get_p4_weighted_by_vif(self):
        p4_influences = P4Influences(function_name="system", p4_influences_as_dict={('no_8x8dct',): np.array([10, 15]), ('no_deblock', 'no_cabac'): np.array([2, 4])})
        features_with_vif = {"('no_cabac',)": 2, "('no_8x8dct',)": 5, "('no_cabac', 'no_deblock')": 8.5}
        p4_influences_expected = P4Influences(function_name="system", p4_influences_as_dict={('no_8x8dct',): np.array([8.75, 13.125]), ('no_deblock', 'no_cabac'): np.array([1.2770, 2.55500])})

        p4_influences_result = self.data_handler.get_weighted_influences_by_vif(p4_influences, features_with_vif)

        np.testing.assert_equal(p4_influences_result, p4_influences_expected)


    def test_calculate_vif_for_feature(self):
        feature_1 = Feature(option1="no_8x8dct", option2="no_deblock", influence=np.array([10, 15]))
        feature_2 = Feature(option1="no_cabac", influence=np.array([10, 15]))
        feature_3 = Feature(option1="no_deblock", option2="no_cabac", influence=np.array([10, 15]))
        features_with_vif = {"('no_8x8dct', 'no_deblock')": 10, "('no_cabac',)": 2, "('no_cabac', 'no_deblock')": 8.5}

        vif_result_1 = self.data_handler.calculate_vif_weight_for_feature(feature_1, features_with_vif)
        vif_result_2 = self.data_handler.calculate_vif_weight_for_feature(feature_2, features_with_vif)
        vif_result_3 = self.data_handler.calculate_vif_weight_for_feature(feature_3, features_with_vif)

        self.assertAlmostEqual(vif_result_1, 0.5, places=3)
        self.assertAlmostEqual(vif_result_2, 0.98, places=3)
        self.assertAlmostEqual(vif_result_3, 0.639, places=3)


if __name__ == '__main__':
    unittest.main()
