import copy
import math
import random
from typing import Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from active_bayesify.utils.dtos.model_result import ModelResult
from active_bayesify.utils.general.data_reader import DataReader
from active_bayesify.utils.models.feature import Feature
from active_bayesify.utils.models.p4_influences import P4Influences


class DataHandler:

    def __init__(self, data_reader: DataReader, option_names: list[str],
                 y_column_names: list[str] = ["taskID", "method", "energy"]):
        self.data_reader = data_reader
        self.option_names = option_names
        self.option_names_with_id = ["taskID"] + option_names
        self.option_names_with_id_and_method = self.option_names_with_id + ["method"]
        self.y_column_names = y_column_names

    def split_data(self, data_frame: pd.DataFrame, train_init_size: int, train_al_size: int, random_state: int,
                   shuffle: bool = True) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits function data into 10 instances of the same taskID for every function on initial training set, 75% for active learning and remaining for testing and their y counterparts.

        :param random_state: random state to pass to splitting function.
        :param shuffle: if split should be random.
        :param data_frame: pandas DataFrame containing the data of the function.
        :return: X_train_init, X_train_al, X_test, y_train_init, y_train_al, y_test.
        """
        train_init_ids, remaining_ids = train_test_split(data_frame["taskID"].unique(), train_size=train_init_size, shuffle=shuffle, random_state=random_state)
        train_init = data_frame[data_frame["taskID"].isin(list(train_init_ids))]
        remaining = data_frame[data_frame["taskID"].isin(list(remaining_ids))]
        train_al_ids, test_ids = train_test_split(remaining["taskID"].unique(), train_size=train_al_size, shuffle=shuffle, random_state=random_state)
        train_al = remaining[remaining["taskID"].isin(list(train_al_ids))]
        test = remaining[remaining["taskID"].isin(list(test_ids))]

        X_train_init = train_init[self.option_names_with_id_and_method]
        X_train_al = train_al[self.option_names_with_id_and_method]
        X_test = test[self.option_names_with_id_and_method]

        y_train_init = train_init[self.y_column_names]
        y_train_al = train_al[self.y_column_names]
        y_test = test[self.y_column_names]

        return X_train_init, X_train_al, X_test, y_train_init, y_train_al, y_test

    def randomly_sample_new_instances(self, X_pool: pd.DataFrame, y_pool: pd.DataFrame, number_of_instances: int,
                                      random_state: int, merge_on: str = "taskID") -> \
            Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly chooses given number of instances from given X and y pool.

        :param random_state: used for randomly sampling from X_pool.
        :param X_pool: pandas DataFrame to sample X instances from.
        :param y_pool: pandas DataFrame to sample Y instances from.
        :param number_of_instances: Integer specifying how many instances to sample.
        :param merge_on: determines on what key field X and y should be sampled.
        :return: tuple of pandas DataFrames containing sampled instances.
        """
        X_instances = self.get_rows_grouped_by_taskID(X_pool).sample(n=number_of_instances, random_state=random_state).sort_values(by=[merge_on])
        y_instances = y_pool[y_pool[merge_on].isin(X_instances[merge_on])].sort_values(by=[merge_on])
        X_instances = X_pool[X_pool["taskID"].isin(list(X_instances["taskID"]))]

        return X_instances, y_instances

    def actively_sample_new_instances(self, X_pool: pd.DataFrame, y_pool: pd.DataFrame, X_train: pd.DataFrame,
                                      p4_influences_list: list[P4Influences], n_features_to_be_included: int,
                                      random_state: int, p4_influences_average: P4Influences,
                                      weight_by_functions: bool = True, weight_by_vif: bool = False,
                                      functions_with_weights: dict[str, float] = None, batch_size: int = 5,
                                      key_column: str = "taskID") -> Tuple[pd.DataFrame, pd.DataFrame, dict, P4Influences, P4Influences, dict[str, float]]:
        """
        Actively chooses given number of instances from given X and y pool based on most uncertain features to include
        and chooses values for the features so that X_train becomes and stays balanced in this respect.

        :param p4_influences_average:
        :param weight_by_functions:
        :param random_state: used to randomly sample from X_pool when active sampling does not find any instances.
        :param weight_by_vif:
        :param functions_with_weights:
        :param X_pool: pandas DataFrame to sample X instances from.
        :param y_pool: pandas DataFrame to sample Y instances from.
        :param X_train: use for comparing to choose new feature value based on values that are already in training set.
        :param p4_influences:
        :param key_column: determines on what key field X and y should be sampled.
        :return: tuple of pandas DataFrames containing sampled instances, dict with features that were set to "on" with their values.
        """
        if weight_by_functions:
            p4_influences_weighted_by_functions = self.get_weighted_p4_influences_by_function_runtime(copy.deepcopy(p4_influences_list), functions_with_weights)
        else:
            p4_influences_weighted_by_functions = copy.deepcopy(p4_influences_average)

        features_with_vif = self.calculate_vif_per_feature(X_train, p4_influences_weighted_by_functions.get_features())

        if weight_by_vif:
            p4_influences_weighted_by_functions_and_vif = self.get_weighted_influences_by_vif(p4_influences_weighted_by_functions, features_with_vif)
        else:
            p4_influences_weighted_by_functions_and_vif = p4_influences_weighted_by_functions

        features_with_high_uncertainty = p4_influences_weighted_by_functions_and_vif.get_features_sorted_by_uncertainty()[:batch_size]

        features_to_be_included = self.order_and_fill_feature_list(X_pool, X_train, features_with_high_uncertainty,
                                                                   n_features_to_be_included, random_state)

        if len(features_to_be_included) == 0: # TODO: irgendeine dieser Funktionen added random samples wenn unter 5 samples hier gefunden wurden, dass nochmal expliziter in den Funktionsnamen machen
            X_instances, y_instances = self.randomly_sample_new_instances(X_pool, y_pool,
                                                                                             n_features_to_be_included,
                                                                                             random_state, key_column)
            return X_instances, y_instances, {}, p4_influences_weighted_by_functions, p4_influences_weighted_by_functions_and_vif, features_with_vif

        X_instances, features_set_to_on_with_values = self.choose_unique_new_instances_based_on_training_pool(X_pool,
                                                                                                              X_train,
                                                                                                              features_to_be_included,
                                                                                                              random_state)

        y_instances = self.get_associated_y_instances(y_pool, X_instances)

        X_instances = X_pool[X_pool[key_column].isin(list(X_instances[key_column]))]

        return X_instances.sort_values(key_column), y_instances.sort_values(key_column), features_set_to_on_with_values, p4_influences_weighted_by_functions, p4_influences_weighted_by_functions_and_vif, features_with_vif

    def calculate_vif_per_feature(self, X_train: pd.DataFrame, features: list[Feature]) -> dict[str, float]:
        """
        Calculates variance inflation factor (VIF) for given features.

        :param X_train: pandas DataFrame to calculate VIF for.
        :param features: list of features to retrieve options from.
        :return: list of options with VIF.
        """
        X_train = X_train.drop_duplicates(subset=["taskID"])
        features_with_vif = {}

        for feature in features:
            if feature.is_interaction():
                cat_dict = {} # TODO: extract this part
                cat_val = 0
                for val1 in X_train[feature.get_option1()].unique():
                    for val2 in X_train[feature.get_option2()].unique():
                        cat_dict[(val1, val2)] = cat_val
                        cat_val += 1

                X_train["merged_options"] = X_train.apply(lambda x: cat_dict[(x[feature.get_option1()], x[feature.get_option2()])], axis=1)
                option_names = [option for option in self.option_names + ["merged_options"] if option not in feature.get_options()]

                if len(option_names) < 2: # if feature consists of all available options, VIF is not defined
                    features_with_vif[str(feature)] = np.nan
                    continue

                feature_vif = variance_inflation_factor(X_train[option_names].values, list(X_train[option_names].columns).index("merged_options"))
            else:
                feature_vif = variance_inflation_factor(X_train[self.option_names].values, list(X_train[self.option_names].columns).index(feature.get_option1()))

            if feature_vif == np.nan:
                feature_vif = 1
            elif feature_vif == math.inf: # happens OLS regression reports perfect collinearity
                feature_vif = 100

            features_with_vif[str(feature)] = feature_vif

        return features_with_vif

    def normalize_dict(self, dictionary: dict[object, float]):
        """
        Normalizes given dictionary with an epsilon value to assign to the lowest value.

        :param dictionary: to normalize.
        :return: normalized dictionary.
        """
        if len(dictionary.keys()):
            dictionary[list(dictionary.keys())[0]] = 1
            return dictionary

        max_vif = max(dictionary.values())
        min_vif = min(dictionary.values())
        epsilon = 0.001
        for key in dictionary.keys():
            dictionary[key] = max((dictionary[key] - min_vif) / (max_vif - min_vif), epsilon)

        return dictionary

    def get_associated_y_instances(self, y_pool: pd.DataFrame, X_instances: pd.DataFrame, key_column: str = "taskID") -> pd.DataFrame:
        """
        Gets y instances for given key from given X instances.

        :param y_pool: to get y instances from.
        :param X_instances: to get key values from.
        :param key_column: key column.
        :return:
        """
        y_instances =  y_pool[y_pool[key_column].isin(list(X_instances[key_column]))]
        return y_instances

    def choose_unique_new_instances_based_on_training_pool(self, X_pool: pd.DataFrame, X_train: pd.DataFrame,
                                                           features_to_be_included: list[Feature], random_state: int) -> Tuple[pd.DataFrame, dict]:
        """
        Chooses a new row from given X pool with the given features set the value that is the less dominant in X_train.

        :param random_state:
        :param X_pool: to choose instances from.
        :param X_train: get value counts from.
        :param features_to_be_included: features to be set to "on" in new instances.
        :return: newly sampled X instances, dict with features that were set to "on" with their values.
        """
        X_instances = pd.DataFrame(columns=X_pool.columns)
        X_instances = X_instances.astype({column: X_pool[column].dtype for column in X_pool.columns})

        features_set_to_on_with_values = {}

        already_chosen_task_ids = []
        for feature in features_to_be_included:
            new_row, taskID = self.chose_row_with_new_task_id(X_pool, X_train, already_chosen_task_ids, feature,
                                                              random_state)

            if taskID in already_chosen_task_ids:
                continue

            features_set_to_on_with_values = self.add_feature_set_least_frequent_value_with_value(feature, features_set_to_on_with_values, new_row, taskID)

            X_instances = pd.concat([X_instances, new_row], ignore_index=True)
            already_chosen_task_ids.append(taskID)

        return X_instances, features_set_to_on_with_values

    def add_feature_set_least_frequent_value_with_value(self, feature: Feature, feature_set_to_on, new_row, taskID):
        if feature.is_interaction():
            first_option, second_option = feature.get_options()
            feature_set_to_on[taskID] = {"feature": feature, "first_value": new_row[first_option].iloc[0], "second_value": new_row[second_option].iloc[0]}
        else:
            feature_set_to_on[taskID] = {"feature": feature, "first_value": new_row[feature.option1].iloc[0], "second_value": None}

        return feature_set_to_on

    def chose_row_with_new_task_id(self, X_pool: pd.DataFrame, X_train: pd.DataFrame,
                                   already_chosen_task_ids: list[int], feature: Feature,
                                   random_state: int) -> Tuple[pd.DataFrame, int]:
        """
        Tries to find a row with feature set to "on" and with a task id that is not already in given list of task ids. If fails fallback to random row.

        :param random_state:
        :param X_pool: to sample row from.
        :param X_train: get value counts from.
        :param already_chosen_task_ids: list of already seen task ids.
        :param feature: should be set to "on" in row.
        :return: sampled row and it's taskID.
        """
        available_rows = self.get_configs_with_feature_set_to_least_frequent_value(X_pool, X_train, feature)

        if self.all_already_chosen_ids_are_in_available_rows(already_chosen_task_ids, available_rows):
            new_row = X_pool[~X_pool["taskID"].isin(already_chosen_task_ids)].sample(n=1, random_state=random_state)
            taskID = self.get_rows_task_id(new_row)
        else:
            new_row = available_rows[~available_rows["taskID"].isin(already_chosen_task_ids)].sample(n=1, random_state=random_state)
            taskID = self.get_rows_task_id(new_row)

        return new_row, taskID

    def get_rows_task_id(self, row: pd.DataFrame) -> int:
        """
        Gets taskID of row.

        :param row: to get taskID for.
        :return: taskID.
        """
        return int(row["taskID"].iloc[0])

    def all_already_chosen_ids_are_in_available_rows(self, task_ids: list[int],
                                                     data: pd.DataFrame) -> bool:
        """
        Checks whether given ids are present in given data frame.

        :param task_ids: list of task ids.
        :param data: to be checked if ids exist here.
        :return:
        """
        return all([id in task_ids for id in list(data["taskID"])])

    def order_and_fill_feature_list(self, X_pool: pd.DataFrame, X_train: pd.DataFrame,
                                    features_to_be_included: list[Feature], n_features_to_be_included: int,
                                    random_state: int) -> list[Feature]:
        """
        Orders feature list to begin with feature with the fewest configs to sample from to avoid feature with more configs to take away this configs.
        For all feature where no configs exist, the list gets filled with other feature.

        :param random_state:
        :param X_pool:
        :param X_train:
        :param features_to_be_included: list of features to be included.
        :return: ordered list of features to be included.
        """
        random.seed(random_state)

        # sort options to start looking for valid configs for option with the lowest config count
        available_samples_per_feature = {str(feature): self.count_configs_with_feature_set_to_least_frequent_value(X_pool, X_train, feature) for feature in
                                         features_to_be_included}
        features_to_be_included = sorted(available_samples_per_feature,
                                         key=lambda entry: available_samples_per_feature[entry], reverse=False)
        features_to_be_included = [Feature(feature_as_string_comma_seperated=feature) for feature in features_to_be_included if
                                   self.count_configs_with_feature_set_to_least_frequent_value(X_pool, X_train, Feature(feature_as_string_comma_seperated=feature)) > 0]

        # adds randomly already present features to list to fill up on features that got sorted out
        n_features_left_to_include = n_features_to_be_included - len(features_to_be_included)

        features_to_be_included_filled = features_to_be_included.copy()

        if n_features_left_to_include == n_features_to_be_included:
            return []

        for _ in range(n_features_left_to_include):
            # TODO: wenn hier nur noch wenige features mit configs im set sind, kann es sein, dass die schleife auch die letzten configs der überbleibenden features entfernt und dann gäbe es hier wieder index out of range exception
            features_to_be_included_filled.append(random.choice(features_to_be_included))

        return features_to_be_included_filled

    def count_configs_with_feature_set_to_least_frequent_value(self, X_pool: pd.DataFrame, X_train: pd.DataFrame, feature: Feature) -> int:
        """
        Counts configs with given option set to "on" in given pool.

        :param X_pool: to check for configs set to on.
        :param X_train: get value counts from.
        :param feature: feature to be checked.
        :return: true if more than 0 configs are found, false else.
        """
        return self.get_configs_with_feature_set_to_least_frequent_value(X_pool, X_train, feature).shape[0]

    def get_configs_with_feature_set_to_least_frequent_value(self, X_pool: pd.DataFrame, X_train: pd.DataFrame, feature: Feature) -> pd.DataFrame:
        """
        Returns configs from pool with given feature set to the least frequent value in training set.

        :param X_pool: to return configs from.
        :param X_train: get value counts from.
        :param feature: feature to be checked.
        :return: DataFrame where option is set to "on".
        """
        if feature.is_interaction():
            option1, option2 = feature.get_options()
            least_frequent_value_option1, least_frequent_value_option2 = self.get_least_frequent_feature_value(X_train, feature)
            return self.get_rows_grouped_by_taskID(X_pool, (X_pool[option1] == least_frequent_value_option1) & (X_pool[option2] == least_frequent_value_option2))
        else:
            least_frequent_value_option = self.get_least_frequent_feature_value(X_train, feature)
            return self.get_rows_grouped_by_taskID(X_pool, X_pool[feature.get_option1()] == least_frequent_value_option)

    def get_least_frequent_feature_value(self, X_train: pd.DataFrame, feature: Feature) -> int:
        """
        Returns least frequent value for given feature.

        :param X_train: to get value counts from.
        :param feature: feature to get values for.
        :return: least frequent value.
        """
        if feature.is_interaction() is False:
            return X_train[feature.get_option1()].value_counts().index[-1]

        return X_train[feature.get_options_as_list()].value_counts().index[-1]

    def get_rows_grouped_by_taskID(self, data: pd.DataFrame, filter = None):
        """
        Groups rows by taskID and return first row for every taskID. Use this function when analyzing on function level.

        :param data:
        :param filter:
        :return:
        """
        if filter is None:
            filter = [True for _ in range(len(data))]

        return data[filter].groupby("taskID").head(1)

    def drop_used_instances(self, original_set: pd.DataFrame, used_instances: pd.DataFrame,
                            merge_on: list[str]) -> pd.DataFrame:
        """
        Drops already processed instances form set.

        :param original_set: pandas DataFrame to remove instances from.
        :param used_instances: pandas DataFrame with used instances.
        :param merge_on: string column name to merge datasets on.
        :return: original pandas DataFrame without used instances.
        """
        merged_set = pd.merge(original_set, used_instances, on=merge_on, how="outer", indicator=True)
        thinned_set = merged_set.loc[merged_set["_merge"] == "left_only"].drop("_merge", axis=1)

        return thinned_set

    def add_new_trainings_data(self, X_train_old: pd.DataFrame, X_train_new: pd.DataFrame, y_train_old: pd.DataFrame,
                               y_train_new: pd.DataFrame) -> \
            Tuple[Union[DataFrame, Series], Union[DataFrame, Series]]:
        """
        Extends given X and y pandas DataFrames with given new instances.

        :param X_train_old: X pandas DataFrame to be extended.
        :param X_train_new: X pandas DataFrame to add.
        :param y_train_old: y pandas DataFrame to be extended.
        :param y_train_new: y pandas DataFrame to add.
        :return: X and y pandas DataFrames.
        """
        return pd.concat([X_train_old, X_train_new], ignore_index=True), pd.concat([y_train_old, y_train_new],
                                                                                   ignore_index=True)

    def get_total_runtime(self, data: pd.DataFrame = None, time_column: str = "time") -> float:
        """
        Calculates total runtime over all rows in data.

        :return: total runtime as float.
        """
        return data[time_column].sum()

    def filter_functions(self, function_names: list[str], total_runtime: float, runtime_threshold: float, function_data_all: pd.DataFrame = None, time_column: str = "time") -> list[str]:
        """
        Returns set of new function that pass the filter.

        :param function_names: functions to be filtered.
        :param total_runtime: system's total runtime.
        :param runtime_threshold: percentage threshold.
        :return: filtered function names.
        """
        new_function_names = []

        for function in function_names:
            if function_data_all is None:
                function_data = self.data_reader.read_in_data(function)
            else:
                function_data = function_data_all[function_data_all["method"] == function]

            if self.is_function_filtered_out(function_data, total_runtime, runtime_threshold, time_column):
                continue

            new_function_names.append(function)

        return new_function_names

    def is_function_filtered_out(self, function_data: pd.DataFrame, total_runtime: float,
                                 runtime_threshold: float, time_column: str = "time") -> bool:
        """
        Tests whether a function does not pass the filter.

        :param runtime_threshold: percentage threshold.
        :param function_data: the function's data.
        :param total_runtime: system's total runtime.
        :return: False if function passes filter, True else.
        """
        function_runtime = np.sum(function_data[time_column])

        if function_runtime / total_runtime < runtime_threshold:
            return True

        return False

    def write_model_results_to_data_frame(self, data: pd.DataFrame, model_results: list[ModelResult]) -> pd.DataFrame:
        """
        Adds selected features from lasso, ridge and p4 and evaluation results to given DataFrame.

        :param data: DataFrame to append data to.
        :param model_results: dictionary with results from models.
        :return: data frame with appended row.
        """
        for model_result in model_results:
            new_row = pd.Series(model_result.as_dict())
            data = self.append_row(data, new_row)

        return data

    def prepare_model_results(self, repetition: int, iteration: int, model_names: list[str], influence_dicts: list[P4Influences], mapes: list[float], function_name: str = "system") -> list[ModelResult]:
        """
        Prepares model results for writing to data frame.

        :param repetition: repetition number.
        :param iteration: iteration number.
        :param model_names: names of models.
        :param influence_dicts: influence dictionaries.
        :param mapes: mean absolute percentage errors.
        :param function_name: name of function.
        :return: list of model results.
        """
        model_results = []

        for model_name, influences, mape in zip(model_names, influence_dicts, mapes):

            if influences is None:
                continue

            for feature in influences.get_features():

                if model_name == "p4":
                    min_influence = feature.get_influence()[0]
                    max_influence = feature.get_influence()[1]
                    influence = np.mean([min_influence, max_influence])
                else:
                    if feature.get_influence() == 0.:
                        continue
                    influence = feature.get_influence()
                    min_influence = np.nan
                    max_influence = np.nan

                model_results.append(
                    ModelResult(model_name, repetition, iteration, feature, mape, influence, min_influence,
                                max_influence, function_name))

        return model_results

    def append_row(self, data_frame: pd.DataFrame, new_row: pd.Series) -> pd.DataFrame:
        """
        Appends a pandas Series to given DataFrame.

        :param data_frame: DataFrame to be extended.
        :param new_row: Series to add to DataFrame.
        :return: concatenated DataFrame.
        """
        return pd.concat([data_frame, new_row.to_frame().T], ignore_index=True).astype(data_frame.dtypes)

    def get_average_p4_influences(self, p4_influences_list: list[P4Influences], n_functions: int,
                                  function_name: str = "system") -> P4Influences:
        """
        Calculates average influence over features in function's influence dictionaries.

        :param p4_influences_list: list of influence dictionaries.
        :param n_functions: number of functions.
        :param function_name: name of function.
        :return: average influence dictionary.
        """
        new_feature_list = []

        for p4_influences in p4_influences_list:
            for feature in p4_influences.get_features():
                is_in_list, idx = self.feature_identified_by_name_in_list(feature, new_feature_list)

                if is_in_list:
                    new_feature_list[idx].set_influence(new_feature_list[idx].get_influence() + feature.get_influence() / n_functions)
                else:
                    feature.set_influence(feature.get_influence() / n_functions)
                    new_feature_list.append(feature)

        return P4Influences(function_name=function_name, p4_influences_as_feature_list=new_feature_list)

    def get_weighted_p4_influences_by_function_runtime(self, p4_influences_list: list[P4Influences], functions_with_weights: dict[str, float], function_name: str = "system") -> P4Influences:
        """
        Calculates weighted influence over features in function's influence dictionaries.

        :param p4_influences_list: list of influence dictionaries.
        :param functions_with_weights: dictionary with function names and weights.
        :param function_name: name of function.
        :return: weighted influence dictionary.
        """
        new_feature_list = []

        for p4_influences in p4_influences_list:
            for feature in p4_influences.get_features():
                is_in_list, idx = self.feature_identified_by_name_in_list(feature, new_feature_list)

                if is_in_list:
                    new_feature_list[idx].set_influence(new_feature_list[idx].get_influence() + feature.get_influence() * functions_with_weights[p4_influences.get_function_name()])
                else:
                    feature.set_influence(feature.get_influence() * functions_with_weights[p4_influences.get_function_name()])
                    new_feature_list.append(feature)

        return P4Influences(function_name=function_name, p4_influences_as_feature_list=new_feature_list)

    def feature_identified_by_name_in_list(self, feature: Feature, feature_list: list[Feature]) -> Union[Tuple[bool, int], Tuple[bool, None]]:
        """
        Checks if feature is in list of features. Strips values from feature and compares only option names.

        :param feature: feature to check.
        :param feature_list: list of features.
        :return: tuple with boolean and index of feature in list. False and None if feature is not in list.
        """
        feature_with_only_option_names = Feature(option1=feature.get_option1(), option2=feature.get_option2())
        new_influences_with_only_option_names = [Feature(option1=feature.get_option1(), option2=feature.get_option2())
                                                 for feature in feature_list]
        if feature_with_only_option_names in new_influences_with_only_option_names:
            return True, new_influences_with_only_option_names.index(feature_with_only_option_names)
        else:
            return False, None

    def get_weighted_influences_by_vif(self, p4_influences_weighted_by_functions: P4Influences, features_with_vif: dict[str, float]):
        """
        Weights p4 influences by variance inflation factor.

        :param p4_influences_weighted_by_functions:
        :param features_with_vif:
        :return:
        """
        p4_influences_weighted_by_vif = copy.deepcopy(p4_influences_weighted_by_functions)

        for feature in p4_influences_weighted_by_vif.get_features():
            feature_vif = self.calculate_vif_weight_for_feature(feature, features_with_vif)

            feature.set_influence(np.round(feature.get_influence() * feature_vif, 3))

        return p4_influences_weighted_by_vif

    def calculate_vif_weight_for_feature(self, feature: Feature, features_with_vif: dict[str, float]):
        feature_vif = features_with_vif[str(feature)]
        epsilon = 0.0001
        feature_vif_weight = max(-0.005 * feature_vif ** 2 + 1, epsilon)
        return feature_vif_weight


