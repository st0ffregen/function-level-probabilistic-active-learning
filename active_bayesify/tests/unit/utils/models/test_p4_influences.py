import unittest
import numpy as np

from active_bayesify.utils.models.feature import Feature
from active_bayesify.utils.models.p4_influences import P4Influences


class P4InfluencesTest(unittest.TestCase):

    def test_init_with_numpy_arrays(self):
        p4_influences_output = {("ref", "rc_lookahead"): np.array([0, 10]), ("rc_lookahead", "threads"): np.array([0, 10]), ("rc_lookahead", "ref"): np.array([5, 15]), ("ref",): np.array([0, 10])}
        p4_influences_expected = [
            Feature(option1="rc_lookahead", option2="ref", influence=np.array([2.5, 12.5])),
            Feature(option1="rc_lookahead", option2="threads", influence=np.array([0, 10])),
            Feature(option1="ref", influence=np.array([0, 10])),
        ]

        p4_influences = P4Influences("test", p4_influences_output)

        self.assertEqual(p4_influences.function_name, "test")
        self.assertEqual(p4_influences.influences, p4_influences_expected)

    def test_init_with_integers(self):
        p4_influences_output = {("ref", "rc_lookahead"): 5, ("rc_lookahead", "threads"): 5, ("rc_lookahead", "ref"): 10, ("ref",): 5}
        p4_influences_expected = [
            Feature(option1="rc_lookahead", option2="ref", influence=7.5),
            Feature(option1="rc_lookahead", option2="threads", influence=5),
            Feature(option1="ref", influence=5),
        ]

        p4_influences = P4Influences("test", p4_influences_output)

        self.assertEqual(p4_influences.function_name, "test")
        self.assertEqual(p4_influences.influences, p4_influences_expected)

    def test_get_influences_sorted_by_uncertainty_with_feature_influeces_as_floats_raise_exception(self):
        p4_influences = P4Influences(function_name="func", p4_influences_as_feature_list=[
            Feature(option1="rc_lookahead", option2="ref", influence=7.5),
        ])

        result = p4_influences.get_features_sorted_by_uncertainty()

        self.assertTrue(isinstance(result, type(NotImplemented)))

    def test_get_influences_sorted_by_uncertainty_descending(self):
        p4_influences = P4Influences(function_name="func", p4_influences_as_feature_list=[
            Feature(option1="rc_lookahead", option2="ref", influence=np.array([10, 20])),
            Feature(option1="ref", option2="threads", influence=np.array([14, 16])),
        ])

        result = p4_influences.get_features_sorted_by_uncertainty()

        np.testing.assert_equal(result[0].get_influence(), np.array([10, 20]))
        np.testing.assert_equal(result[1].get_influence(), np.array([14, 16]))

    def test_get_influences_sorted_by_uncertainty_descending(self):
        p4_influences = P4Influences(function_name="func", p4_influences_as_feature_list=[
            Feature(option1="rc_lookahead", option2="ref", influence=np.array([10, 20])),
            Feature(option1="ref", option2="threads", influence=np.array([14, 16])),
        ])

        result = p4_influences.get_features_sorted_by_uncertainty(desc=False)

        np.testing.assert_equal(result[0].get_influence(), np.array([14, 16]))
        np.testing.assert_equal(result[1].get_influence(), np.array([10, 20]))


if __name__ == '__main__':
    unittest.main()
