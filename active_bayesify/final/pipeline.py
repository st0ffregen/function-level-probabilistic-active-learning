import copy
import multiprocessing as mp
from typing import Tuple, Any, Union
import argparse

import numpy as np
import pandas as pd
from bayesify.pairwise import PyroMCMCRegressor, P4Preprocessing

from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.utils.general.data_reader import DataReader
from active_bayesify.utils.general.logger import Logger
from active_bayesify.utils.models.p4_influences import P4Influences
from active_bayesify.utils.pipeline.data_handler import DataHandler
from active_bayesify.utils.pipeline.emc_sampler import EMCSampler
from active_bayesify.utils.pipeline.standard_models import StandardModels


class ActiveBayesify:

    feature_model = "x264.xml"

    def __init__(self, system_name, batch_size, option_names: list[str]):
        self.option_names = option_names
        self.option_names_with_id = ["taskID"] + self.option_names
        self.option_names_with_id_and_method = ["method"] + self.option_names_with_id
        config_parser = ConfigParser(system_name)

        self.path_to_data = config_parser.get_path_with_system_name("Data")
        self.path_to_results = config_parser.get_path_with_system_name("Results")
        self.path_to_logs = config_parser.get("Paths", "Logs")
        self.runtime_threshold = 0.01
        self.batch_size = batch_size

        self.distance_based_sampling_wrapper = None # DistanceBasedSamplingWrapper()

        self.logger = Logger(config_parser).get_logger()
        self.data_reader = DataReader(config_parser)
        self.data_handler = DataHandler(self.data_reader, self.option_names)
        self.emc_sampler = EMCSampler(self.data_handler, self.logger, self.option_names, self.train_and_evaluate_p4)

        self.standard_models = StandardModels(self.option_names, self.data_handler)

        feature_model = None # ET.parse(self.path_to_feature_model + self.feature_model)
        self.splc2py_sampler = None # Sampler(feature_model, "docker")

    def evaluate_model(self, model: PyroMCMCRegressor, preprocessor: P4Preprocessing, X_test: pd.DataFrame,
                       y_test: pd.DataFrame) -> Tuple[float, Any]:
        """
        Makes model predict on given X_test and evaluate predictions by calculating MAPE and PSIS information. TODO: more doc here

        :param model: to make prediction.
        :param preprocessor: to transform X_test to obtain new features.
        :param X_test: to make predictions on.
        :param y_test: to compare predictions to.
        :return: MAPE and PSIS information. TODO: see above
        """
        X_test_numpy = X_test[self.option_names].to_numpy()
        y_test_numpy = y_test["energy"].to_numpy()
        new_features = preprocessor.transform(X_test_numpy)

        prediction_with_n_samples_for_each_x = model.predict(new_features, n_samples=1000)
        prediction = np.mean(prediction_with_n_samples_for_each_x, axis=0)
        prediction_score = self.standard_models.calculate_mape(y_test_numpy, prediction)

        elpd_psis = np.mean(model.loo(pointwise=True).data)

        return prediction_score, elpd_psis

    # noinspection PyTypeChecker
    def preprocess_data(self, X_train: np.array, y_train: np.array, feature_names: list[str], preprocessor: P4Preprocessing = None) -> Tuple[
        P4Preprocessing, pd.DataFrame,
        list[str]]:
        """
        Applies P4 preprocessing. Retrieves transformed features and associated features names.
        :param X_train: to be transformed.
        :param y_train: needed in preprocessing.
        :param feature_names: old features names.
        :return: preprocessor instance, transformed X_train, new features names.
        """
        if preprocessor is None:
            preprocessor = P4Preprocessing()  # TODO: macht das normalization? Sonst muss ich das vorher noch machen

        new_features = preprocessor.fit_transform(X_train, y_train, feature_names=feature_names)
        new_features_names = preprocessor.feature_names_out
        # TODO: rename new_features
        return preprocessor, new_features, new_features_names

    def train_p4(self, X_train: pd.DataFrame, y_train: np.array, new_feature_names: list[str], random_state: int) -> Tuple[
        PyroMCMCRegressor,
        list[str]]:
        """
        Trains model on given data.

        :param random_state:
        :param X_train: new features gained from preprocessing to train model on.
        :param y_train: to train model on.
        :param new_feature_names: new features names gained from preprocessing.
        :return: model instance, new features as string array.
        """
        reg = PyroMCMCRegressor()
        mcmc_cores = 1

        # TODO: entweder lasso findet nicht genug oder hier kommt ein fehler den ich nicht verstehe
        reg.fit(X_train, y_train, mcmc_samples=1000, mcmc_tune=2000, feature_names=new_feature_names,
                mcmc_cores=mcmc_cores, random_key=random_state)

        return reg, new_feature_names

    def train_and_evaluate_p4(self, function_name: str, repetition: int, iteration: int, X_train: pd.DataFrame,
                              X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, random_state: int) -> \
            Union[tuple[None, None, None, None], tuple[Any, float, PyroMCMCRegressor, P4Preprocessing]]:
        """
        Trains and evaluates model afterwards. Returns MAPE and influences dictionary.

        :param random_state:
        :param function_name: function for which data P4 is trained on.
        :param iteration: used for logging.
        :param repetition: used for logging.
        :param X_train: to train model on.
        :param X_test: to test model on.
        :param y_train: to train model on.
        :param y_test: true y labels.
        :return: influences, dict MAPE score or None if error occurs.
        """
        try:
            X_train_numpy = X_train[self.option_names].to_numpy()
            y_train_numpy = y_train["energy"].to_numpy()

            preprocessor, new_features, new_feature_names = self.preprocess_data(X_train_numpy, y_train_numpy,
                                                                                 self.option_names)
            model, _ = self.train_p4(new_features, y_train_numpy, new_feature_names, random_state)
            influences = P4Influences(function_name, model.coef_ci(0.95)["influences"])

            self.logger.info(
                f"{function_name}:{iteration}:{repetition}: model was trained successfully!")
            prediction_score, _ = self.evaluate_model(model, preprocessor, X_test, y_test)
            self.logger.info(f"{function_name}:{iteration}:{repetition}: model has score of {prediction_score}")
        except (AssertionError, KeyError, RuntimeError) as e:  # TODO: explain errors in comment
            self.logger.warning(f"{function_name}:{iteration}:{repetition}: error was thrown: {e}")
            return None, None, None, None
        except Exception as e:
            self.logger.error(f"{function_name}:{iteration}:{repetition}: critical error was thrown: {e}")
            raise e

        return influences, prediction_score, model, preprocessor

    def get_or_create_existing_results(self, file_name: str, columns: list[str]) -> pd.DataFrame:
        """
        Gets existing results from given file. If no data exists returns an empty DataFrame.

        :param file_name: to retrieve data from.
        :param columns: columns to initialize DataFrame with.
        :return: filled or empty DataFrame.
        """
        try:
            return pd.read_csv(f"{self.path_to_results}{file_name}.csv")
        except FileNotFoundError:
            self.data_reader.create_directory_if_not_exists(self.path_to_results)
            empty_frame = pd.DataFrame(columns=columns)
            empty_frame.to_csv(f"{self.path_to_results}{file_name}.csv", index=False)
            return empty_frame
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=columns)

    def run_models(self, args):
        X_test, X_train, function_name, repetition, iteration, results_frame_columns, y_test, y_train, random_state = args

        function_column = "method"
        X_train = X_train[X_train[function_column] == function_name]
        X_test = X_test[X_test[function_column] == function_name]
        y_train = y_train[y_train[function_column] == function_name]
        y_test = y_test[y_test[function_column] == function_name]

        X_train_numpy = X_train[self.option_names].to_numpy()
        y_train_numpy = y_train["energy"].to_numpy()

        if X_train.shape[0] < 3:
            self.logger.warning(f"{function_name}:{iteration}:{repetition}: models were not trained because of too few data in trainings set! |X_train|: {X_train.shape[0]}")
            return None, None, None, None

        if X_test.shape[0] < 3:
            self.logger.warning(f"{function_name}:{iteration}:{repetition}: models were not trained because of too few data in test set! |X_test|: {X_test.shape[0]}")
            return None, None, None, None

        lasso_influences, lasso_mape = self.standard_models.run_lasso(function_name, X_train_numpy, y_train_numpy, 1.0,
                                                                      X_test, y_test, random_state)
        ridge_influences, ridge_mape = self.standard_models.run_ridge(function_name, X_train_numpy, y_train_numpy, 1.0,
                                                                      X_test, y_test, random_state)
        p4_influences, p4_mape, p4_model, p4_preprocessor = self.train_and_evaluate_p4(function_name, repetition,
                                                                                       iteration, X_train, X_test,
                                                                                       y_train, y_test, random_state)
        model_results = self.data_handler.prepare_model_results(repetition, iteration,
                                                                ["lasso", "ridge", "p4"],
                                                                [lasso_influences, ridge_influences,
                                                                 p4_influences], [lasso_mape, ridge_mape,
                                                                                  p4_mape], function_name)
        results = self.data_handler.write_model_results_to_data_frame(pd.DataFrame(columns=results_frame_columns), model_results)
        return p4_influences, results, p4_model, p4_preprocessor

    def filter_functions(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        """
        Retrieve top functions with runtime threshold and filter data.

        :param data: pandas data frame to retrieve functions and to be filtered.
        :return:
        """
        function_names = list(data["method"].unique())
        total_runtime = self.data_handler.get_total_runtime(data)
        function_names = self.data_handler.filter_functions(function_names, total_runtime, self.runtime_threshold, data)
        data = data[data["method"].isin(function_names)]

        return data, function_names

    def actively_learn_on_function_level(self, data_file_name: str, results_file_name: str,
                                         weight_by_functions=True, weight_by_vif=False,
                                         active_with_emc=False, is_real_emc=False, is_debug=False):
        """
        Runs naive active learning approach on function level.
        :param weight_by_functions:
        :param is_debug:
        """
        self.logger.info(
            f"{0}:{0}:{None}: Iteratively train on function level with active learning.")

        data = self.data_reader.read_in_data(data_file_name)
        data, function_names = self.filter_functions(data)
        functions_with_weights = self.calculate_normalized_function_weights(data, function_names)

        self.run_learning(results_file_name, data, function_names, True, weight_by_functions=weight_by_functions, functions_with_weights=functions_with_weights,
                          weight_by_vif=weight_by_vif, active_with_emc=active_with_emc, is_real_emc=is_real_emc,
                          is_debug=is_debug)

    def actively_learn_on_system_level(self, data_file_name: str, results_file_name: str, is_debug=False):
        """
        Runs active learning on system level.
        :param is_debug:
        """
        self.logger.info(
            f"{0}:{0}:system: Iteratively train whole system with active learning.")

        data = self.data_reader.read_in_data(data_file_name)
        function_names = ["system"]
        functions_with_weights = self.calculate_normalized_function_weights(data, function_names)

        self.run_learning(results_file_name, data, function_names, True, weight_by_functions=False, functions_with_weights=functions_with_weights,
                          is_debug=is_debug)


    def randomly_learn_on_system_level(self, data_file_name: str, results_file_name: str, is_debug=False):
        """
        Runs random learning approach on system level.
        :param is_debug:
        """
        self.logger.info(
            f"{0}:{0}:system: Iteratively train whole system with random learning.")

        data = self.data_reader.read_in_data(data_file_name)
        function_names = ["system"]

        self.run_learning(results_file_name, data, function_names, False, is_debug=is_debug)

    def randomly_learn_on_function_level(self, data_file_name: str, results_file_name: str, is_debug=False):
        """
        Runs random learning approach on function level.
        :param is_debug:
        """
        self.logger.info(
            f"{0}:{0}:{None}: Iteratively train on function level with random learning.")

        data = self.data_reader.read_in_data(data_file_name)
        data, function_names = self.filter_functions(data)

        self.run_learning(results_file_name, data, function_names, False, is_debug=is_debug)

    def run_learning(self, results_file_name: str, data: pd.DataFrame, function_names: list[str], is_active: bool,
                     weight_by_functions=True, functions_with_weights: dict[str, float] = None,
                     weight_by_vif=False, active_with_emc=False, is_real_emc=False, is_debug=False):
        """
        Runs active or random learning.
        :param weight_by_functions:
        :param is_debug:
        :param is_real_emc:
        :param active_with_emc:
        """
        train_init_size = 10
        train_al_size = 90 if len(data["taskID"].unique()) > 130 else 50 # brotli does only provide on system level around 80 samples after filtering our short running functions

        chosen_configs_file_name, features_vif_file_name, results_file_name, weighted_and_average_p4_results_file_name = self.init_result_frames_file_names(
            results_file_name)

        existing_results, results, existing_chosen_configs, chosen_configs, existing_weighted_and_average_p4_results, weighted_and_average_p4_results, existing_features_results_vif, features_vif_results = self.init_result_frames(results_file_name, chosen_configs_file_name, weighted_and_average_p4_results_file_name, features_vif_file_name)
        repetition = self.determine_current_repetition(existing_results)
        random_state = repetition

        X_train_init, X_train_al, X_test, y_train_init, y_train_al, y_test = self.data_handler.split_data(data,
                                                                                                          train_init_size,
                                                                                                          train_al_size,
                                                                                                          random_state)

        chosen_configs = self.add_chosen_configs_information(X_train_init[self.option_names_with_id], chosen_configs, is_active, repetition,
                                                             train_init_size)

        p4_influences_list, results, p4_models, p4_preprocessors = self.run_models_for_functions(X_train_init,
                                                                                                 y_train_init, X_test,
                                                                                                 y_test, results,
                                                                                                 repetition,
                                                                                                 train_init_size,
                                                                                                 function_names,
                                                                                                 random_state,
                                                                                                 is_debug is False)

        X_train = X_train_init
        y_train = y_train_init

        max_iteration = int(train_al_size / self.batch_size)
        for i in range(1, max_iteration + 1):
            iteration = train_init_size + i * self.batch_size
            self.logger.info(f"{repetition}:{i}:system: Iteration: {i}/{max_iteration}")

            features_set_to_on_with_values = None
            n_functions = len(function_names)
            p4_influences_average = self.data_handler.get_average_p4_influences(p4_influences_list, n_functions)

            if is_active:
                if active_with_emc:
                    X_instances, y_instances = self.emc_sampler.actively_sample_new_instances_with_parameter_change(
                        X_train_al, y_train_al, X_train, y_train, X_test, y_test, p4_influences_average, p4_models,
                        p4_preprocessors, list(functions_with_weights.keys())[0], random_state, self.batch_size, is_real_emc,
                        is_debug is False)

                    features = [feature for features in p4_influences_list for feature in features.get_features()]
                    features_with_vif = self.data_handler.calculate_vif_per_feature(X_train, features)
                    p4_influences_weighted = None
                    p4_influences_weighted_by_vif = None
                else:
                    X_instances, y_instances, features_set_to_on_with_values, p4_influences_weighted, p4_influences_weighted_by_vif, features_with_vif = self.data_handler.actively_sample_new_instances(
                        X_train_al, y_train_al, X_train, p4_influences_list, self.batch_size, random_state,
                        p4_influences_average,
                        weight_by_functions=weight_by_functions, weight_by_vif=weight_by_vif,
                        functions_with_weights=functions_with_weights, batch_size=self.batch_size)
            else:

                features = [feature for features in p4_influences_list for feature in features.get_features()]
                features_with_vif = self.data_handler.calculate_vif_per_feature(X_train, features)

                X_instances, y_instances = self.data_handler.randomly_sample_new_instances(
                    X_train_al, y_train_al, number_of_instances=self.batch_size, random_state=random_state)
                p4_influences_weighted = None
                p4_influences_weighted_by_vif = None

            weighted_and_average_p4_results = self.write_weighted_and_average_p4_influences_to_data_frame(
                repetition,
                iteration,
                p4_influences_average,
                p4_influences_weighted,
                p4_influences_weighted_by_vif,
                weighted_and_average_p4_results)

            features_vif_results = self.write_features_vif_to_data_frame(is_active, repetition, iteration, features_with_vif, features_vif_results)


            chosen_configs = self.add_chosen_configs_information(X_instances[self.option_names_with_id], chosen_configs, is_active, repetition,
                                                                 iteration, features_set_to_on_with_values)

            X_train, y_train = self.data_handler.add_new_trainings_data(X_train, X_instances, y_train, y_instances)

            p4_influences_list, results, p4_models, p4_preprocessors = self.run_models_for_functions(X_train, y_train,
                                                                                                     X_test, y_test,
                                                                                                     results,
                                                                                                     repetition,
                                                                                                     iteration,
                                                                                                     function_names,
                                                                                                     random_state,
                                                                                                     is_debug is False)

            X_train_al, y_train_al = self.drop_used_instances(X_instances, X_train_al, y_instances, y_train_al)

        self.collect_and_save_results_and_meta_information(is_active, chosen_configs_file_name, existing_chosen_configs,
                                                           chosen_configs, features_vif_file_name,
                                                           existing_features_results_vif, features_vif_results,
                                                           results_file_name, existing_results, results,
                                                           weighted_and_average_p4_results_file_name,
                                                           existing_weighted_and_average_p4_results,
                                                           weighted_and_average_p4_results)

        self.logger.info(f"done")

    def init_result_frames_file_names(self, results_file_name):
        chosen_configs_file_name = f"{results_file_name}_chosen_configs"
        weighted_and_average_p4_results_file_name = f"{results_file_name}_weighted_and_average_p4_results"
        features_vif_file_name = f"{results_file_name}_features_vif"
        results_file_name = f"{results_file_name}_results"
        return chosen_configs_file_name, features_vif_file_name, results_file_name, weighted_and_average_p4_results_file_name

    def collect_and_save_results_and_meta_information(self, is_active: bool, chosen_configs_file_name: str,
                                                      existing_chosen_configs: pd.DataFrame,
                                                      chosen_configs: pd.DataFrame, features_vif_file_name: str,
                                                      existing_features_results_vif: pd.DataFrame,
                                                      features_vif_results: pd.DataFrame, results_file_name: str,
                                                      existing_results: pd.DataFrame, results: pd.DataFrame,
                                                      weighted_and_average_p4_results_file_name: str,
                                                      existing_weighted_and_average_p4_results: pd.DataFrame,
                                                      weighted_and_average_p4_results: pd.DataFrame) -> None:
        """
        Collects and saves the results and chosen configs, weight and vif information of the current repetition.
        """
        results["is_active"] = is_active
        results = pd.concat([existing_results, results])

        chosen_configs = pd.concat([existing_chosen_configs, chosen_configs])

        weighted_and_average_p4_results["is_active"] = is_active
        weighted_and_average_p4_results = pd.concat(
            [existing_weighted_and_average_p4_results, weighted_and_average_p4_results])

        features_vif_results = pd.concat([existing_features_results_vif, features_vif_results])

        self.logger.info(f"write results to file")
        results.to_csv(f"{self.path_to_results}{results_file_name}.csv", index=False)

        self.logger.info(f"write chosen configs information to file")
        chosen_configs.to_csv(f"{self.path_to_results}{chosen_configs_file_name}.csv", index=False)

        self.logger.info(f"write weighted and average p4 results to file")
        weighted_and_average_p4_results.to_csv(f"{self.path_to_results}{weighted_and_average_p4_results_file_name}.csv",
                                               index=False)
        self.logger.info(f"write features with vif to file")
        features_vif_results.to_csv(f"{self.path_to_results}{features_vif_file_name}.csv", index=False)

    def add_chosen_configs_information(self, instances: pd.DataFrame, chosen_configs: pd.DataFrame, is_active: bool,
                                       repetition: int, iteration: int, features_set_to_on: dict = None,
                                       key_column: str = "taskID") -> pd.DataFrame:
        """
        Adds information gained from given data frame about chosen configs to given frame.

        :param instances:
        :param chosen_configs:
        :param is_active:
        :param repetition:
        :param iteration:
        :param key_column:
        :param features_set_to_on:
        :return:
        """
        configs_dict_list = instances.groupby(key_column).head(1).to_dict("records")
        general_info = {"is_active": is_active, "repetition": repetition, "iteration": iteration,
                 "feature": None, "first_option_value": None, "second_option_value": None}

        rows = []
        for config_dict in configs_dict_list:
            config_task_id = config_dict[key_column]
            if features_set_to_on is not None and config_task_id in features_set_to_on.keys():
                general_info["feature"] = features_set_to_on[config_task_id]["feature"]
                general_info["first_option_value"] = features_set_to_on[config_task_id]["first_value"]
                general_info["second_option_value"] = features_set_to_on[config_task_id]["second_value"]
            rows.append(pd.Series({**general_info, **config_dict}))

        chosen_configs = pd.concat([chosen_configs, pd.DataFrame(rows)], ignore_index=True)

        return chosen_configs

    def drop_used_instances(self, X_instances: pd.DataFrame, X_train_al: pd.DataFrame, y_instances: pd.DataFrame, y_train_al: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Removes used instances from X and y pool.

        :param X_instances: used X instances.
        :param X_train_al: X pool to drop instances from.
        :param y_instances: used y instances.
        :param y_train_al: y pool to drop instances from.
        :return: thinned X and y pool.
        """
        X_train_al = self.data_handler.drop_used_instances(X_train_al, X_instances,
                                                           self.option_names_with_id_and_method)
        y_train_al = self.data_handler.drop_used_instances(y_train_al, y_instances, ["taskID", "energy", "method"])

        return X_train_al, y_train_al

    def run_models_for_functions(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_test: pd.DataFrame, results: pd.DataFrame, repetition: int, iteration: int,
                                 function_names: list[str], random_state: int, is_multiprocessing=True) -> Tuple[list[P4Influences], pd.DataFrame, list[PyroMCMCRegressor], list[P4Preprocessing]]:
        """
        Loops through list of functions and runs lasso, ridge and p4 for every function.

        :param is_multiprocessing:
        :param X_train: data to be trained on.
        :param y_train: data to be trained on.
        :param X_test: data to be tested on.
        :param y_test: data to be tested on.
        :param results: results from previous runs.
        :param repetition: used for logging.
        :param iteration: used for logging.
        :param function_names: list of functions. Can be set to "system" for system-wide training.
        :return: list of p4 influences and results.
        """
        p4_influences_of_all_models = []
        function_names = function_names if is_multiprocessing is True else function_names[:2] # only for debugging
        args_list = [(X_test, X_train, function_name, repetition, iteration, results.columns, y_test, y_train, random_state) for
                function_name in function_names]
        p4_models, p4_preprocessors = {}, {}

        if is_multiprocessing:
            # solves logging problem when multithreading, see https://pythonspeed.com/articles/python-multiprocessing/
            # TODO: add exception handling on outer pool level
            with mp.get_context("spawn").Pool(processes=min((mp.cpu_count() - 1), 20)) as pool:
                pool_results = pool.map(self.run_models, args_list)

            for pool_result in pool_results:
                p4_influences, model_results, p4_model, p4_preprocessor = pool_result

                if p4_influences is None:
                    continue

                p4_models[p4_influences.get_function_name()] = p4_model
                p4_preprocessors[p4_influences.get_function_name()] = p4_preprocessor
                p4_influences_of_all_models.append(p4_influences)
                results = pd.concat([results, model_results], ignore_index=True)

        else:
            for args in args_list:
                p4_influences, model_results, p4_model, p4_preprocessor = self.run_models(args)

                if p4_influences is None:
                    continue

                p4_models[p4_influences.get_function_name()] = p4_model
                p4_preprocessors[p4_influences.get_function_name()] = p4_preprocessor
                p4_influences_of_all_models.append(p4_influences)
                results = pd.concat([results, model_results], ignore_index=True)

        return p4_influences_of_all_models, results, p4_models, p4_preprocessors

    def write_features_vif_to_data_frame(self, is_active: bool, repetition: int, iteration: int, features_with_vif: dict[str, float], features_vif_results: pd.DataFrame) -> pd.DataFrame:
        """
        Writes feature vif to data frame.

        :param repetition: number of repetition.
        :param iteration: number of iteration.
        :param features_with_vif: feature vif as dict.
        :param features_vif_results: data frame to write to.
        :return: data frame with feature vif.
        """
        for feature in features_with_vif.keys():
            features_vif_results = features_vif_results.append(
                {"is_active": is_active, "repetition": repetition, "iteration": iteration, "feature": feature, "vif": features_with_vif[feature]},
                ignore_index=True)

        return features_vif_results

    def write_weighted_and_average_p4_influences_to_data_frame(self, repetition, iteration, p4_influences_average,
                                                               p4_influences_weighted_by_functions, p4_influences_weighted_by_vif,
                                                               weighted_and_average_p4_results_data_frame):
        """
        Writes weighted and average p4 influences to data frame.

        :param repetition: number of repetition.
        :param iteration: number of iteration.
        :param p4_influences_average: p4 influences averaged over all functions.
        :param p4_influences_weighted_by_functions: p4 influences weighted by functions over all functions.
        :param p4_influences_weighted_by_vif: p4 influences weighted by vif over all functions.
        :param weighted_and_average_p4_results_data_frame: data frame to write to.
        :return: data frame with weighted and average p4 influences.
        """
        weighted_p4_influences_by_functions_system_results = self.data_handler.prepare_model_results(repetition, iteration, ["p4"],
                                                                                        [p4_influences_weighted_by_functions],
                                                                                        [np.nan],
                                                                                        "weighted_p4_influences_by_functions_system_results")
        average_p4_influences_system_results = self.data_handler.prepare_model_results(repetition, iteration, ["p4"],
                                                                                       [p4_influences_average],
                                                                                       [np.nan],
                                                                                       "average_p4_influences_system_results")
        weighted_p4_influences_by_vif_system_results = self.data_handler.prepare_model_results(repetition, iteration, ["p4"], [p4_influences_weighted_by_vif], [np.nan], "weighted_p4_influences_by_vif_system_results")
        weighted_and_average_p4_results_data_frame = self.data_handler.write_model_results_to_data_frame(weighted_and_average_p4_results_data_frame, weighted_p4_influences_by_functions_system_results)
        weighted_and_average_p4_results_data_frame = self.data_handler.write_model_results_to_data_frame(weighted_and_average_p4_results_data_frame, average_p4_influences_system_results)
        weighted_and_average_p4_results_data_frame = self.data_handler.write_model_results_to_data_frame(weighted_and_average_p4_results_data_frame, weighted_p4_influences_by_vif_system_results)
        return weighted_and_average_p4_results_data_frame

    def determine_current_repetition(self, existing_results: pd.DataFrame) -> int:
        """
        Retrieves repetition from already existing results or sets it to 0.

        :param existing_results: already existing results.
        :return: repetition number.
        """
        if existing_results.shape[0] > 0:
            repetition = existing_results["repetition"].max() + 1
        else:
            repetition = 0

        return repetition

    def init_result_frames(self, results_file_name: str, chosen_configs_file_name: str, weighted_and_average_p4_results_file_name: str, features_vif_file_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Initializes an empty frame for saving results in and reads in already existing results.

        :param results_file_name: to read in existing results from.
        :param chosen_configs_file_name: to read in existing information about chose configs from.
        :param weighted_and_average_p4_results_file_name: to read in existing weighted and average p4 results from.
        :param features_vif_file_name: to read in existing feature vif from.
        :return: existing_results, results, existing_chosen_configs, chosen_configs, weighted_and_average_p4_results
        """
        results_columns = ["is_active", "model_name", "repetition", "iteration", "function_name", "feature", "influence", "min_influence",
                   "max_influence", "mape"]
        existing_results = self.get_or_create_existing_results(results_file_name, results_columns)
        results = pd.DataFrame(columns=results_columns)

        existing_weighted_and_average_p4_results = self.get_or_create_existing_results(weighted_and_average_p4_results_file_name, results_columns)
        weighted_and_average_p4_results = pd.DataFrame(columns=results_columns)

        chosen_configs_columns = ["is_active", "repetition", "iteration", "feature", "first_option_value", "second_option_value"] + self.option_names_with_id
        existing_chosen_configs = self.get_or_create_existing_results(chosen_configs_file_name, chosen_configs_columns)
        chosen_configs = pd.DataFrame(columns=chosen_configs_columns)

        features_vif_columns = ["is_active", "repetition", "iteration", "feature", "vif"]
        existing_features_vif_results = self.get_or_create_existing_results(features_vif_file_name, features_vif_columns)
        features_vif_results = pd.DataFrame(columns=features_vif_columns)

        return existing_results, results, existing_chosen_configs, chosen_configs, existing_weighted_and_average_p4_results, weighted_and_average_p4_results, existing_features_vif_results, features_vif_results


    def calculate_normalized_function_weights(self, data, function_names, time_column: str = "time"):
        functions_with_runtime = {}
        for function_name in function_names:
            runtime = data[data["method"] == function_name][time_column].sum()

            functions_with_runtime[function_name] = runtime

        normalized_functions = self.data_handler.normalize_dict(functions_with_runtime)
        normalized_functions = {k: v for k, v in sorted(normalized_functions.items(), key=lambda item: item[1])}
        return normalized_functions


def main():
    is_debug = False

    parser = argparse.ArgumentParser(description="Pipeline to run active and random learning with P4.")
    parser.add_argument("--system-name", help="Name of the system to run learning on.", type=str)
    parser.add_argument("--batch_size", help="Batch size for active learning.", type=int, default=5)
    parser.add_argument("--active", help="Run active learning.", action="store_true")
    parser.add_argument("--random", help="Run random learning.", action="store_true")
    parser.add_argument("--system", help="Run system level learning.", action="store_true")
    parser.add_argument("--function", help="Run function level learning.", action="store_true")
    parser.add_argument("--emc", help="Run EMC learning.", action="store_true")
    parser.add_argument("--brute-force-emc", help="Run simulated brute force EMC learning.", action="store_true")
    parser.add_argument("--function-committee", help="Run with function committee weighting.", action="store_true")
    parser.add_argument("--vif", help="Run with VIF weighting.", action="store_true")
    parser.add_argument("--file", help="File to run learning on.", type=str)
    parser.add_argument("--output", help="Output file name.", type=str)
    args = parser.parse_args()

    args = get_arguments() if is_debug else args

    option_names = get_option_names(args)

    pipeline = ActiveBayesify(args.system_name, args.batch_size, option_names)

    if args.active:
        if args.system:
            pipeline.actively_learn_on_system_level(args.file, args.output, is_debug=is_debug)
        elif args.function:
            pipeline.actively_learn_on_function_level(args.file, args.output, weight_by_functions=args.function_committee, weight_by_vif=args.vif,
                                                      active_with_emc=args.emc,
                                                      is_real_emc=args.brute_force_emc is False, is_debug=is_debug)
    elif args.random:
        if args.system:
            pipeline.randomly_learn_on_system_level(args.file, args.output, is_debug)
        elif args.function:
            pipeline.randomly_learn_on_function_level(args.file, args.output, is_debug)


def get_option_names(args):
    config_parser = ConfigParser(args.system_name)
    data_reader = DataReader(config_parser)
    data = data_reader.read_in_data(args.file)
    option_names = [option for option in list(data.columns) if option not in ["taskID", "energy", "time", "method"]]
    return option_names


def get_arguments():
    parser = argparse.ArgumentParser(description="Pipeline to run active and random learning with P4.")
    args = parser.parse_args()
    args.system_name = "x264"
    args.batch_size = 5
    args.active = True
    args.random = False
    args.system = False
    args.function = True
    args.emc = True
    args.brute_force_emc = True
    args.function_committee = False
    args.vif = False
    args.file = "x264_ml_cfg"
    args.output = "test"

    return args


if __name__ == "__main__":
    main()
