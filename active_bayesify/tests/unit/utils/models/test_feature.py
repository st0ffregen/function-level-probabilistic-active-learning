import unittest
import numpy as np

from active_bayesify.utils.models.feature import Feature


class FeatureTest(unittest.TestCase):

    def test_init_with_option(self):
        feature = Feature(option1="threads")

        self.assertEqual(feature.option1, "threads")

    def test_init_with_options(self):
        feature = Feature(option1="threads", option2="ref")

        self.assertEqual(feature.option1, "ref")
        self.assertEqual(feature.option2, "threads")

    def test_init_feature_as_tuple(self):
        feature = Feature(feature_as_tuple=("ref", "threads"))

        self.assertEqual(feature.option1, "ref")
        self.assertEqual(feature.option2, "threads")

    def test_init_feature_as_tuple_inverted(self):
        feature = Feature(feature_as_tuple=("threads", "ref"))

        self.assertEqual(feature.option1, "ref")
        self.assertEqual(feature.option2, "threads")

    def test_init_feature_as_comma_seperated_string(self):
        feature = Feature(feature_as_string_comma_seperated="('threads', 'ref')")

        self.assertEqual(feature.option1, "ref")
        self.assertEqual(feature.option2, "threads")

    def test_init_feature_as_space_seperated_string(self):
        feature = Feature(feature_as_string_space_seperated="a b")

        self.assertEqual(feature.option1, "a")
        self.assertEqual(feature.option2, "b")

    def test_init_feature_as_space_seperated_string(self):
        feature = Feature(feature_as_string_space_seperated="a")

        self.assertEqual(feature.option1, "a")

    def test_init_feature_as_space_seperated_string_inverted(self):
        feature = Feature(feature_as_string_space_seperated="b a")

        self.assertEqual(feature.option1, "a")
        self.assertEqual(feature.option2, "b")

    def test_init_with_values(self):
        feature = Feature(option1="ref", option2="threads", option1_value=10, option2_value=100)

        self.assertEqual(feature.option1_value, 10)
        self.assertEqual(feature.option2_value, 100)

    def test_init_with_values_inverted(self):
        feature = Feature(option1="threads", option2="ref", option1_value=10, option2_value=100)

        self.assertEqual(feature.option1, "ref")
        self.assertEqual(feature.option2, "threads")
        self.assertEqual(feature.option1_value, 100)
        self.assertEqual(feature.option2_value, 10)

    def test_init_with_influence(self):
        feature = Feature(option1="ref", option2="threads", influence=np.array([0, 10]))

        np.testing.assert_equal(feature.influence, np.array([0, 10]))

    def test_is_interaction_is_an_interaction(self):
        feature = Feature(option1="ref", option2="threads")

        self.assertTrue(feature.is_interaction())

    def test_is_interaction_is_no_interaction(self):
        feature = Feature(option1="ref")

        self.assertFalse(feature.is_interaction())

    def test_to_string_single_option(self):
        feature = Feature(option1="ref")

        self.assertEqual(str(feature), "('ref',)")

    def test_to_string_two_options(self):
        feature = Feature(option1="ref", option2="threads")

        self.assertEqual(str(feature), "('ref', 'threads')")

    def test_as_file_name_string_single_option(self):
        feature = Feature(option1="ref")

        self.assertEqual(feature.as_file_name_string(), "ref")

    def test_as_pretty_string_two_options(self):
        feature = Feature(option1="ref", option2="threads")

        self.assertEqual(feature.as_pretty_string(), "(ref, threads)")

    def test_as_pretty_string_single_option(self):
        feature = Feature(option1="ref")

        self.assertEqual(feature.as_pretty_string(), "(ref)")

    def test_as_file_name_string_two_options(self):
        feature = Feature(option1="ref", option2="threads")

        self.assertEqual(feature.as_file_name_string(), "ref_threads")

    def test_get_options(self):
        feature = Feature(option1="ref", option2="threads")

        self.assertTrue(isinstance(feature.get_options(), tuple))

    def test_equals_success(self):
        feature1 = Feature(option1="ref", option2="threads", option1_value=1, option2_value=1)
        feature2 = Feature(option1="ref", option2="threads", option1_value=1, option2_value=1)

        self.assertTrue(feature1 == feature2)

    def test_equals_inverted_success(self):
        feature1 = Feature(option1="ref", option2="threads", option1_value=1, option2_value=1)
        feature2 = Feature(option1="threads", option2="ref", option1_value=1, option2_value=1)

        self.assertTrue(feature1 == feature2)

    def test_equals_fails(self):
        feature1 = Feature(option1="ref", option2="threads", option1_value=1, option2_value=1)
        feature2 = Feature(option1="ref", option2="threads")

        self.assertFalse(feature1 == feature2)


if __name__ == '__main__':
    unittest.main()
