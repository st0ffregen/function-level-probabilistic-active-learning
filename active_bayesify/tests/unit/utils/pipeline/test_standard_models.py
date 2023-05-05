import unittest
from unittest.mock import Mock

from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from active_bayesify.utils.models.feature import Feature
from active_bayesify.utils.models.p4_influences import P4Influences
from active_bayesify.utils.pipeline.data_handler import DataHandler
from active_bayesify.utils.pipeline.standard_models import StandardModels


class StandardModelsTests(unittest.TestCase):

    def setUp(self) -> None:
        self.option_names = ["a", "b"]
        self.feature_names_with_interaction = ["a", "b", "a b", "b a"]
        data = [[0, 0], [0, 1], [1, 0], [1, 1]]

        self.X_train = pd.DataFrame(columns=self.option_names, data=data)
        self.X_test = pd.DataFrame(columns=self.option_names, data=data)
        self.y_train = pd.DataFrame(columns=["energy"], data=[1, 2, 3, 4])
        self.y_test = pd.DataFrame(columns=["energy"], data=[2, 3, 4, 5])

        self.standard_models = StandardModels(self.option_names, DataHandler(Mock(), self.option_names))

    def test_run_lasso(self):
        X_train = self.X_train[self.option_names].to_numpy()
        y_train = self.y_train["energy"].to_numpy()

        coefs, mape = self.standard_models.run_lasso("test", X_train, y_train, 1.0, self.X_test, self.y_test, 0)

        self.assertAlmostEqual(mape, 32, places=0)

    def test_run_ridge(self):
        X_train = self.X_train[self.option_names].to_numpy()
        y_train = self.y_train["energy"].to_numpy()

        coefs, mape = self.standard_models.run_ridge("test", X_train, y_train, 1.0, self.X_test, self.y_test, 0)

        self.assertAlmostEqual(mape, 27, places=0)

    def test_run_polynomial_features(self):
        X_train = self.X_train[self.option_names].to_numpy()

        result = self.standard_models.run_polynomial_features(X_train)

        self.assertTrue((result[0] == np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])).all())

    def test_evaluate_standard_model(self):
        X_train = self.X_train[self.option_names].to_numpy()
        y_train = self.y_train["energy"].to_numpy()

        poly_reg = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly_reg.fit_transform(X_train)

        model = linear_model.Lasso(alpha=1.0)
        model.fit(X_train_poly, y_train)

        result = self.standard_models.evaluate_standard_model(model, poly_reg, self.X_test, self.y_test)

        self.assertAlmostEqual(result, 32, places=0)

    def test_parse_model_to_p4_output(self):
        coefs = [0, 2, 4, 3]
        expected = P4Influences("test", p4_influences_as_feature_list=[
            Feature(option1="a", influence=0),
            Feature(option1="b", influence=2),
            Feature(option1="a", option2="b", influence=3.5)
        ])

        result = self.standard_models.parse_model_to_p4_output("test", coefs,
                                                               self.feature_names_with_interaction)

        self.assertEqual(expected, result)

    def test_calculate_mape(self):
        y_true = [10, 5]
        y_pred = [9, 4]

        result = self.standard_models.calculate_mape(y_true, y_pred)

        self.assertAlmostEqual(15, result)



if __name__ == '__main__':
    unittest.main()
