import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from active_bayesify.utils.models.p4_influences import P4Influences
from active_bayesify.utils.pipeline.data_handler import DataHandler


class StandardModels:

    def __init__(self, option_names, data_handler: DataHandler):
        self.option_names = option_names
        self.data_handler = data_handler

    def run_lasso(self, function_name: str, x_train_numpy: np.array, y_train_numpy: np.array, alpha: float,
                  X_test: pd.DataFrame, y_test: pd.DataFrame, random_state: int) -> tuple[P4Influences, float]:
        """
        Run standard model Lasso on data.

        :param function_name:
        :param x_train_numpy: X data to train model on.
        :param y_train_numpy: y data to train model on.
        :param alpha: Ridge's aplha parameter.
        :param X_test: X test data to predict on.
        :param y_test: y test data to predict against.
        :param random_state: random state for model.
        :return: dict with features and coefficients, MAPE, model
        """
        X_poly, feature_names_out, poly_reg = self.run_polynomial_features(x_train_numpy)

        model = linear_model.Lasso(alpha=alpha, max_iter=5000, random_state=random_state)
        model.fit(X_poly, y_train_numpy)

        mape = self.evaluate_standard_model(model, poly_reg, X_test, y_test)

        return self.parse_model_to_p4_output(function_name, model.coef_, feature_names_out), mape

    def run_ridge(self, function_name: str, x_train_numpy: np.array, y_train_numpy: np.array, alpha: float, X_test: pd.DataFrame, y_test: pd.DataFrame, random_state: int) -> tuple[P4Influences, float]:
        """
        Run standard model Ridge on data.

        :param x_train_numpy: X data to train model on.
        :param y_train_numpy: y data to train model on.
        :param alpha: Ridge's aplha parameter.
        :param X_test: X test data to predict on.
        :param y_test: y test data to predict against.
        :param random_state: random state for model.
        :return: dict with features and coefficients, MAPE
        """
        X_poly, feature_names_out, poly_reg = self.run_polynomial_features(x_train_numpy)

        model = linear_model.Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_poly, y_train_numpy)

        mape = self.evaluate_standard_model(model, poly_reg, X_test, y_test)

        return self.parse_model_to_p4_output(function_name, model.coef_, feature_names_out), mape

    def run_polynomial_features(self, x_train_numpy: np.array) -> tuple[np.array, list[str], PolynomialFeatures]:
        """
        Transforms data with PolynomialFeatures.

        :param x_train_numpy: data to be transformed.
        :return: transformed X DataFrame, transformed feature names, PolynomialFeatures instance
        """
        poly_reg = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly_reg.fit_transform(x_train_numpy)
        feature_names_out = poly_reg.get_feature_names_out(self.option_names)
        return X_poly, feature_names_out, poly_reg

    def evaluate_standard_model(self, model: object, poly_preprocessor: PolynomialFeatures, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """
        Transforms X_test array to alculates MAPE for standard models.

        :param model: model to use for prediction.
        :param poly_preprocessor: preprocessor to transform test data with.
        :param X_test: DataFrame holding X test data.
        :param y_test: DataFrame holding y test data.
        :return:
        """
        X_test_numpy = X_test[self.option_names].to_numpy()
        y_test_numpy = y_test["energy"].to_numpy()

        X_test_poly = poly_preprocessor.transform(X_test_numpy)

        prediction = model.predict(X_test_poly)
        mape = self.calculate_mape(y_test_numpy, prediction)

        return mape

    def parse_model_to_p4_output(self, function_name: str, coefs: np.array, feature_names_out: list[str]) -> P4Influences:
        """
        Parses model coefficients and new features names into same format as P4 provides.

        :param function_name: used to pass to P4Influences object.
        :param coefs: model coefficients.
        :param feature_names_out: feature names returned by PolynomialFeatures.
        :return: dictionary with coefficients and features equally structured like P4's dict.
        """
        influences = {}
        for idx, feature in enumerate(feature_names_out):
            if feature == "1": # TODO: what does this case check?
                continue

            split = feature.split(" ")
            if len(split) == 2:
                feature_name = (split[0], split[1])
            else:
                feature_name = (feature,)

            influences[feature_name] = coefs[idx]

        influences = P4Influences(function_name, p4_influences_as_dict=influences)
        return influences

    def calculate_mape(self, y_test: np.array, predictions: np.array) -> np.array:
        """
        Calculates Mean Absolute Percentage Error (not in percentage) over the prediction array.

        :param y_test: pandas DataFrame of the true y data.
        :param predictions: numpy array containing prediction.
        :return: numpy array containing mape.
        """
        prediction_score = mean_absolute_percentage_error(y_test, predictions) * 100
        return prediction_score
