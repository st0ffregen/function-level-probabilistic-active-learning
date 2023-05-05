import math
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable

from active_bayesify.utils.general.config_parser import ConfigParser
from active_bayesify.utils.general.data_reader import DataReader
from active_bayesify.utils.general.logger import Logger
from active_bayesify.utils.models.feature import Feature
from active_bayesify.utils.pipeline.data_handler import DataHandler

sns.set(rc={"figure.figsize": (11, 7)})
sns.set_style("whitegrid", {'axes.grid': False})
sns.set(font_scale=1.2)
#sns.set_palette(sns.color_palette("cubehelix", as_cmap=True))

class ActiveBayesifyEvaluation:

    def __init__(self, system_name):
        config_parser = ConfigParser(system_name)

        self.path_to_data = config_parser.get_path_with_system_name("Data")
        self.path_to_results = config_parser.get_path_with_system_name("Results")
        self.path_to_logs = config_parser.get_path_with_system_name("Logs")
        self.path_to_images = config_parser.get_path_with_system_name("Images")
        self.runtime_threshold = config_parser.getfloat("Pipeline", "RuntimeThreshold")
        self.batch_size = config_parser.getint("Pipeline", "BatchSize")
        self.number_of_repetition_per_function = config_parser.getint("Pipeline", "NumberRepetitions")

        self.logger = Logger(config_parser).get_logger()
        self.data_reader = DataReader(config_parser)
        self.data_handler = DataHandler(self.data_reader, [])


    def plot_model_feature_selection_and_their_influences(self, data: pd.DataFrame, model_names: list[str], function_names: list[str] = ["system"]):
        """
        Plots a bar chart with feature selection frequency and features influence for a given model from a given csv file with the data.
        Treats features with influence equals 0 as not selected features.

        :param file_name: to read data from.
        :param model_name: plot data of this model.
        """

        max_features_in_graph = 20

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)

                if model_data.shape[0] == 0:
                    continue

                n_functions = self.get_number_of_functions(model_data)

                single_feature_filter = model_data["feature"].str.contains(",\)")
                single_features = self.get_number_of_feature_occurrences_and_feature_influence_with_filter(model_data,
                                                                                                           single_feature_filter, max_features_in_graph, n_functions)

                pair_wise_features_filter = ~model_data["feature"].str.contains(",\)")
                pair_wise_features = self.get_number_of_feature_occurrences_and_feature_influence_with_filter(model_data,
                                                                                                              pair_wise_features_filter, max_features_in_graph, n_functions)

                result = {**single_features, **pair_wise_features}

                max_selection_frequency = model_data["iteration"].nunique() * model_data["repetition"].nunique()
                selection_frequency_in_percent = [100 * e["occurrences"] / max_selection_frequency for e in result.values()]

                self.render_bar_chart_with_color_bar(list(result.keys()), selection_frequency_in_percent,
                                                     [abs(e["influences"]) for e in result.values()],
                                                     "selected features", "selection frequency in percent",
                                                     "normalized absolute influence",
                                                     f"influence of selected features by {model_name} for function {function_name}",
                                                     "selected_features_with_influence_by_model_and_function/",
                                                     f"selected_features_with_influence_by_{model_name}_and_function_{function_name}")

    def get_number_of_functions(self, model_data):
        n_functions = len(model_data["function_name"].unique()) if "function_name" in model_data.columns else 1
        return n_functions

    def get_number_of_feature_occurrences_and_feature_influence_with_filter(self, data: pd.DataFrame, fil, max_features_in_graph: int, n_functions: int = 1):
        """
        Creates a dictionary with a feature as a key containing the total feature occurrences and the mean feature influence.

        :param data: to retrieve data from.
        :param fil: predefined filter, usually filters out pair-wise or single features.
        :param max_features_in_graph: limits number of features.
        :return:
        """
        feature_dict = {}
        for feature, _ in data[fil].groupby(["feature"]).size().sort_values(ascending=False).head(max_features_in_graph).items():
            feature_dict[feature] = {
                "occurrences": math.ceil(data[data["feature"] == feature].count()[0] / n_functions),
                "influences": data[data["feature"] == feature]["influence"].mean()
            }

        return feature_dict

    def render_bar_chart_with_color_bar(self, x, y, colors, xlabel, ylabel, cbar_label, title, directory_name,
                                        file_name):
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 7))
        data_color = [x / max(colors) for x in colors]

        my_cmap = plt.cm.get_cmap('viridis')
        colors = my_cmap(data_color)

        ax.bar(x, y, color=colors)
        sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, max(data_color)))
        sm.set_array([])

        cbar = plt.colorbar(sm)
        cbar.set_label(cbar_label, rotation=270, labelpad=25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.autofmt_xdate()
        self.data_reader.create_images_directory_if_not_exists(directory_name)
        plt.savefig(f"{self.path_to_images}{directory_name}{file_name}.png")

    def insert_missing_repetition_iteration_tuples(self, data, overall_data: pd.DataFrame):
        """
        Inserts into a dataframe grouped by repetition and iteration and a value column tuples with value 0 for all missing repetition and iteration tuples.
        Guesses batch size automatically from the other more intact groups.

        :param data: to identify missing tuples in.
        :param overall_data: used to determine iterations. Does not use data because it is prone to more missing tuples.
        :return: data frame where missing repetition and iterations are filled in.
        """

        iterations = [list(iterations) for iterations in list(overall_data.groupby(["repetition"]).agg({"iteration": "unique"})["iteration"])]

        repetitions_with_too_few_iterations = self.get_repetitions_with_too_few_iterations(data)
        estimated_batch_size = self.get_estimated_batch_size(iterations)
        iterations_that_should_exist = self.get_standard_iterations(estimated_batch_size, iterations)

        for rep in repetitions_with_too_few_iterations:
            missing_iterations = list(sorted(set(iterations_that_should_exist) - set(iterations[rep])))
            for missing_iteration in missing_iterations:
                data.loc[(rep, missing_iteration), :] = 0

        data = data.sort_index()

        return data

    def get_standard_iterations(self, estimated_batch_size, iterations):
        first_item = np.min([iteration[0] for iteration in iterations])
        length = np.max([len(iteration) for iteration in iterations])
        iterations_that_should_exist = [first_item + estimated_batch_size * i for i in range(length)]
        return iterations_that_should_exist

    def get_estimated_batch_size(self, iterations):
        estimated_batch_size = np.mean(
            [np.mean([second - first for first, second in zip(iteration, iteration[1:])]) for iteration in iterations])

        return int(estimated_batch_size) if np.isnan(estimated_batch_size) is False else None

    def get_repetitions_with_too_few_iterations(self, data):
        existing_tuples = data.index
        iteration_count_per_repetition = existing_tuples.to_frame(index=False).groupby(["repetition"])["iteration"].agg(
            ["count"])
        mean_iterations = iteration_count_per_repetition.mean()["count"]
        estimated_iteration_number = math.ceil(mean_iterations)
        repetitions_with_too_few_iterations = iteration_count_per_repetition[
            iteration_count_per_repetition["count"] < estimated_iteration_number].index.values
        return repetitions_with_too_few_iterations

    def calculate_average(self, data: list[list[int]]):
        return np.mean(data, axis=0)

    def calculate_total(self, data: list[list[int]]):
        return np.sum(data, axis=0)

    def plot_line_graph(self, data, x_column, y_column, hue_column, title, x_label, y_label, directory_name, file_name,
                        style_column=None, size_column=None, **kwargs):
        """

        :param directory_name:
        :param data:
        :param x_column:
        :param y_column:
        :param hue_column:
        :param title:
        :param x_label:
        :param y_label:
        :param file_name:
        :param style_column:
        :param size_column:
        :param kwargs: Please specify with keys "axis" and "plot" if kwargs are passed to this very function to modify the axes or passed to seaborn plotting function to modify the plot.
        """
        if data.shape[0] == 0:
            return

        plt.clf()

        plot_kwargs = kwargs["plot"] if "plot" in kwargs.keys() else {}
        ax = sns.relplot(data=data, kind="line", x=x_column, y=y_column, hue=hue_column, style=style_column, size=size_column, height=7, aspect=11 / 7, **plot_kwargs)
        if kwargs is not None and "axes" in kwargs.keys():
            ax.set(**kwargs["axes"])
        ax.set(title=title)
        ax.set(xlabel=x_label, ylabel=y_label)

        self.data_reader.create_images_directory_if_not_exists(directory_name)
        plt.savefig(f"{self.path_to_images}{directory_name}{file_name}", bbox_inches="tight")

    def plot_bar_plot(self, data, x_column, y_column, title, x_label, y_label, directory_name, file_name,
                      hue_column=None):
        plt.clf()

        ax = sns.barplot(data=data, x=x_column, y=y_column, hue=hue_column)
        ax.set(title=title)
        ax.set(xlabel=x_label, ylabel=y_label)
        self.data_reader.create_images_directory_if_not_exists(directory_name)
        plt.savefig(f"{self.path_to_images}{directory_name}{file_name}", bbox_inches="tight")

    def plot_scatter_plot(self, data, x_column, y_column, hue_column, title, x_label, y_label, directory_name,
                          file_name, style_column=None, has_axhline=False, **kwargs):
        if data.shape[0] == 0:
            return

        plt.clf()

        if data[hue_column].dtype == bool:
            marker_map = {True: "X", False: "o"}
            color_map = {True: "tab:orange", False: "tab:blue"}
        else:
            marker_map = None
            color_map = None

        plot_kwargs = kwargs["plot"] if "plot" in kwargs.keys() else {}
        ax = sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue_column, style=style_column, markers=marker_map, palette=color_map, **plot_kwargs)
        if kwargs is not None and "axes" in kwargs.keys():
            ax.set(**kwargs["axes"])

        if has_axhline:
            plt.axhline(y=0, color='black', linestyle='--')
        ax.set(title=title)
        ax.set(xlabel=x_label, ylabel=y_label)
        self.data_reader.create_images_directory_if_not_exists(directory_name)
        plt.savefig(f"{self.path_to_images}{directory_name}{file_name}", bbox_inches="tight")


    def plot_feature_selection_count_and_influence_over_active_learning_process(self, data: pd.DataFrame, model_names: list[str], function_names: list[str] = ["system"]):
        """
        Plots a graph showing the number of selected features over the time of the active sampling process for the models.

        :param file_name: name of csv file with data.
        """

        feature_selection = pd.DataFrame(columns=["model_name", "repetition", "iteration", "count", "is_active"])

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)

                if model_data.shape[0] == 0:
                    continue

                n_functions = self.get_number_of_functions(model_data)

                feature_count_grouped_by_repetition_and_iteration = model_data.groupby(["repetition", "iteration"])[
                    "feature"].agg(["count"])

                feature_count_grouped_by_repetition_and_iteration = self.insert_missing_repetition_iteration_tuples(
                    feature_count_grouped_by_repetition_and_iteration, data)

                feature_count_grouped_by_repetition_and_iteration.loc[:, "count"] = np.ceil(feature_count_grouped_by_repetition_and_iteration["count"] / n_functions)

                feature_count_grouped_by_repetition_and_iteration = feature_count_grouped_by_repetition_and_iteration.reset_index()
                feature_count_grouped_by_repetition_and_iteration.loc[:, "model_name"] = model_name
                feature_count_grouped_by_repetition_and_iteration.loc[:, "is_active"] = model_data["is_active"].max()

                feature_selection = pd.concat([feature_selection, feature_count_grouped_by_repetition_and_iteration], ignore_index=True)

            feature_selection = self.sort_for_plotting(feature_selection)
            self.plot_line_graph(feature_selection, "iteration", "count", "model_name",
                                 f"feature count for function {function_name}", "number of sampled configurations",
                                 "feature count", "feature_selection_count_over_active_sampling_for_function/",
                                 f"feature_selection_count_over_active_sampling_for_function_{function_name}.png",
                                 "is_active")

    def plot_influence_of_feature_selection_over_active_learning_process(self, data: pd.DataFrame, model_names: list[str], function_names: list[str] = ["system"]):
        """
        Plots a graph showing the influence of selected features over the time of the active (here random) sampling process for the models lasso and p4.

        :param file_name: name of csv file with data.
        """
        feature_selection = pd.DataFrame(columns=["model_name", "repetition", "iteration", "mean", "is_active"])

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)

                if model_data.shape[0] == 0:
                    continue

                feature_influence_grouped_by_repetition_and_iteration = model_data.groupby(["repetition", "iteration"]).agg(
                    {f"influence": "mean"})

                feature_influence_grouped_by_repetition_and_iteration = feature_influence_grouped_by_repetition_and_iteration.rename(columns={f"influence": "mean"})

                feature_influence_grouped_by_repetition_and_iteration = self.insert_missing_repetition_iteration_tuples(
                    feature_influence_grouped_by_repetition_and_iteration, data)

                feature_influence_grouped_by_repetition_and_iteration = feature_influence_grouped_by_repetition_and_iteration.reset_index()
                feature_influence_grouped_by_repetition_and_iteration.loc[:, "model_name"] = model_name
                feature_influence_grouped_by_repetition_and_iteration.loc[:, "is_active"] = model_data["is_active"].max()

                feature_selection = pd.concat([feature_selection, feature_influence_grouped_by_repetition_and_iteration], ignore_index=True)

            feature_selection = self.sort_for_plotting(feature_selection)
            self.plot_line_graph(feature_selection, "iteration", "mean", "model_name",
                                 f"mean feature influence for function {function_name}",
                                 "number of sampled configurations", "feature influence mean", "feature_influence_for_function/",
                                 f"feature_influence_for_function_{function_name}.png", "is_active", **{"axes": {"ylim": (-0.5, 0.5)}})

    def get_model_and_active_zip(self, model_names: list[str]):
        return zip(list(sorted(model_names * 2)), [True, False] * len(model_names))

    def plot_influence_of_all_features_and_individual_top_features_for_repetitions(self, data: pd.DataFrame,
                                                                                   model_names: list[str] = ["p4"],
                                                                                   n_top_uncertain_features: int = 5,
                                                                                   function_names: list[str] = ["system"]):
        """
        Plots one graph each repetition showing the influence of all features and individually of the top n features that were selected at the fist iteration over the time of the active sampling process.
        """
        top_n_column_name = f"top_{n_top_uncertain_features}_uncertain_feature"

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)
                repetitions = list(data["repetition"].unique())

                for repetition in repetitions:
                    model_data_for_rep_i = model_data[model_data["repetition"] == repetition]

                    if model_data_for_rep_i.shape[0] == 0:
                        continue

                    model_data_for_rep_i.loc[:, "influence"] = model_data_for_rep_i["influence"].abs()

                    model_data_for_rep_i[top_n_column_name] = "all_features"

                    min_iteration = model_data_for_rep_i["iteration"].min()

                    top_uncertain_features = list(model_data_for_rep_i[model_data_for_rep_i["iteration"] == min_iteration].sort_values(by=["influence"], ascending=[False]).head(n_top_uncertain_features)["feature"])
                    model_data_for_rep_i_with_top_n_features = model_data_for_rep_i.copy(deep=True)
                    model_data_for_rep_i_with_top_n_features = model_data_for_rep_i_with_top_n_features[model_data_for_rep_i_with_top_n_features["feature"].isin(top_uncertain_features)]
                    model_data_for_rep_i_with_top_n_features[top_n_column_name] = model_data_for_rep_i_with_top_n_features["feature"]

                    model_data_for_rep_i = pd.concat([model_data_for_rep_i, model_data_for_rep_i_with_top_n_features], ignore_index=True)
                    model_data_for_rep_i.loc[:,"model_name"] = model_name
                    model_data_for_rep_i.loc[:,"is_active"] = model_data["is_active"].max()

                    model_data_for_rep_i = self.sort_for_plotting(model_data_for_rep_i)

                    # TODO: here model_name is ignored
                    self.plot_line_graph(model_data_for_rep_i, "iteration", "influence", top_n_column_name,
                                         f"mean feature influence for function {function_name} in repetition {repetition}",
                                         "number of sampled configurations", "feature influence mean", "feature_influence_with_top_n_features_for_function_and_repetition/",
                                         f"feature_influence_with_top_n_features_for_function_{function_name}_and_repetition_{repetition}.png",
                                         **{"plot": {"ci": None}, "axes": {"ylim": (0, 2)}})

    def plot_influence_of_all_features_and_average_top_features(self, data: pd.DataFrame,
                                                                model_names: list[str],
                                                                n_top_uncertain_features: int,
                                                                function_names: list[str] = ["system"]):
        """
        Plots a graph showing the influence of all features and the average of the top n features over the time of the active sampling process.

        :param function_names:
        :param file_name: name of csv file with data.
        """
        top_n_column_name = f"is_top_{n_top_uncertain_features}_uncertain_feature"
        feature_influences = pd.DataFrame(columns=["model_name", "repetition", "iteration", "mean", "is_active", top_n_column_name])

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)

                if model_data.shape[0] == 0:
                    continue

                model_data.loc[:, "influence"] = model_data["influence"].abs()
                feature_influence_grouped_by_repetition_and_iteration = model_data.groupby(["repetition", "iteration"]).agg({f"influence": "mean"})

                feature_influence_grouped_by_repetition_and_iteration = self.insert_missing_repetition_iteration_tuples(feature_influence_grouped_by_repetition_and_iteration, data)

                feature_influence_grouped_by_repetition_and_iteration.loc[:, "is_top_n_uncertain_feature"] = False

                top_uncertain_features = model_data.sort_values(by=["repetition", "iteration", "influence"], ascending=[True, True, False]).groupby(["repetition", "iteration"]).head(n_top_uncertain_features).groupby(["repetition", "iteration"]).agg(
                    {f"influence": "mean"})

                top_uncertain_features.loc[:, "is_top_n_uncertain_feature"] = True

                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features = pd.concat([feature_influence_grouped_by_repetition_and_iteration, top_uncertain_features])

                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features = feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.rename(columns={f"influence": "mean"})

                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features = feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.reset_index()
                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.loc[:, "model_name"] = model_name
                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.loc[:, "is_active"] = model_data["is_active"].max()

                feature_influences = pd.concat([feature_influences, feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features], ignore_index=True)

            feature_influences = self.sort_for_plotting(feature_influences)

            self.plot_line_graph(feature_influences, "iteration", "mean", "model_name",
                                 f"mean feature influence for function {function_name}",
                                 "number of sampled configurations", "feature influence mean", "feature_influence_with_top_n_features_for_function/",
                                 f"feature_influence_with_top_n_features_for_function_{function_name}.png",
                                 style_column="is_active", size_column=top_n_column_name)

    def get_model_data_for_function(self, data, function_name, is_active, model_name):
        model_data = self.get_model_data(data, is_active, model_name)
        if function_name == "system" and "system" not in model_data["function_name"].unique():
            model_data = model_data
        else:
            model_data = model_data[model_data["function_name"] == function_name]
        return model_data

    def get_model_data(self, data, is_active, model_name, additional_filter = None):
        fil = ((data["model_name"] == model_name) & (data[f"influence"] != 0.) & (data["is_active"] == is_active))
        model_data = data[fil]

        if additional_filter is not None:
            model_data = model_data[additional_filter]

        return model_data


    def sort_for_plotting(self, data, column_names: list[str] = ["model_name", "repetition", "iteration", "is_active"], ):
        column_names = [column_name for column_name in column_names if column_name in list(data.columns)] # TODO: warning if not in column names
        data = data.sort_values(by=column_names)
        return data

    def plot_ci_interval_width_of_all_features_and_individual_top_features_for_repetition(self, data: pd.DataFrame,
                                                                                   model_names: list[str] = "p4",
                                                                                   n_top_uncertain_features: int = 5,
                                                                                   function_names: list[str] = ["system"]):
        """
        Plots one graph each repetition showing the ci interval width of all features and individually of the top n features that were selected at the first iteration over the time of the active sampling process.

        :param file_name: name of csv file with data.
        """
        top_n_column_name = f"top_{n_top_uncertain_features}_uncertain_feature"

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)
                repetitions = list(data["repetition"].unique())

                for repetition in repetitions:
                    model_data_for_rep_i = model_data[model_data["repetition"] == repetition]

                    if model_data_for_rep_i.shape[0] == 0:
                        continue

                    model_data_for_rep_i.loc[:, "ci_interval_width"] = abs(model_data_for_rep_i["max_influence"] - model_data_for_rep_i["min_influence"])

                    model_data_for_rep_i.loc[:, top_n_column_name] = "all_features"
                    min_iteration = model_data_for_rep_i["iteration"].min()

                    top_uncertain_features = list(
                        model_data_for_rep_i[model_data_for_rep_i["iteration"] == min_iteration].sort_values(
                            by=["ci_interval_width"], ascending=[False]).head(n_top_uncertain_features)["feature"])
                    model_data_for_rep_i_with_top_n_features = model_data_for_rep_i.copy(deep=True)
                    model_data_for_rep_i_with_top_n_features = model_data_for_rep_i_with_top_n_features[
                        model_data_for_rep_i_with_top_n_features["feature"].isin(top_uncertain_features)]
                    model_data_for_rep_i_with_top_n_features[top_n_column_name] = model_data_for_rep_i_with_top_n_features[
                        "feature"]

                    model_data_for_rep_i = pd.concat([model_data_for_rep_i, model_data_for_rep_i_with_top_n_features], ignore_index=True)
                    model_data_for_rep_i.loc[:,"model_name"] = model_name
                    model_data_for_rep_i.loc[:,"is_active"] = model_data["is_active"].max()
                    model_data_for_rep_i = self.sort_for_plotting(model_data_for_rep_i)

                    # TODO: here model name is ignored
                    self.plot_line_graph(model_data_for_rep_i, "iteration", "ci_interval_width", top_n_column_name,
                                         f"mean ci interval width for function {function_name} in repetition {repetition}",
                                         "number of sampled configurations", "ci interval width", "feature_ci_interval_width_with_top_n_features_for_function_and_repetition/",
                                         f"feature_ci_interval_width_with_top_n_features_for_function_{function_name}_and_repetition_{repetition}.png",
                                         **{"plot": {"ci": None}, "axes": {"ylim": (0, 2)}})


    def plot_ci_interval_width_of_all_features_and_top_features_over_active_learning_process(self, data: pd.DataFrame, model_names: list[str], n_top_uncertain_features: int, function_names: list[str] = ["system"]):
        """
        Plots a graph showing the ci interval width of all features and of the top n features over the time of the active sampling process.

        :param file_name: name of csv file with data.
        """
        feature_ci_interval_width = pd.DataFrame(columns=["model_name", "repetition", "iteration", "mean", "is_active", "is_top_n_uncertain_feature"])

        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)

                if model_data.shape[0] == 0:
                    continue

                model_data.loc[:, "ci_interval_width"] = abs(model_data["max_influence"] - model_data["min_influence"])
                feature_influence_grouped_by_repetition_and_iteration = model_data.groupby(["repetition", "iteration"]).agg(
                    {f"ci_interval_width": "mean"})

                feature_influence_grouped_by_repetition_and_iteration = self.insert_missing_repetition_iteration_tuples(feature_influence_grouped_by_repetition_and_iteration, data)

                feature_influence_grouped_by_repetition_and_iteration.loc[:, "is_top_n_uncertain_feature"] = False

                top_uncertain_features = model_data.sort_values(by=["repetition", "iteration", "ci_interval_width"], ascending=[True, True, False]).groupby(["repetition", "iteration"]).head(n_top_uncertain_features).groupby(["repetition", "iteration"]).agg(
                    {f"ci_interval_width": "mean"})

                top_uncertain_features.loc[:, "is_top_n_uncertain_feature"] = True

                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features = pd.concat([feature_influence_grouped_by_repetition_and_iteration, top_uncertain_features])

                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features = feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.rename(columns={f"ci_interval_width": "mean"})

                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features = feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.reset_index()
                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.loc[:, "model_name"] = model_name
                feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features.loc[:, "is_active"] = model_data["is_active"].max()

                feature_ci_interval_width = pd.concat([feature_ci_interval_width, feature_influence_grouped_by_repetition_and_iteration_with_top_uncertain_features], ignore_index=True)

            feature_ci_interval_width = self.sort_for_plotting(feature_ci_interval_width)
            self.plot_line_graph(feature_ci_interval_width, "iteration", "mean", "model_name",
                                 f"mean ci interval width for function {function_name}",
                                 "number of sampled configurations", "ci interval width", "feature_ci_interval_width_with_top_n_features_for_function/",
                                 f"feature_ci_interval_width_with_top_n_features_for_function_{function_name}.png",
                                 style_column="is_active", size_column="is_top_n_uncertain_feature")


    def plot_mape_for_function(self, data: pd.DataFrame, function_names: list[str] = ["system"]):
        for function_name in function_names:
            if function_name == "system" and "system" not in data["function_name"].unique():
                function_data = data
            else:
                function_data = data[data["function_name"] == function_name]

            function_data = self.sort_for_plotting(function_data)
            self.plot_line_graph(function_data, "iteration", "mape", "model_name",
                                 f"MAPE over sampling process for function {function_name}",
                                 "number of sampled configurations", "MAPE", "mape_over_active_sampling_for_function/",
                                 f"mape_over_active_sampling_for_function_{function_name}.png", "is_active")

    def is_feature_set_to_on(self, row, chosen_configs_data):
        feature = Feature(feature_as_string_comma_seperated=row["feature"])
        chosen_configs_data_same_rep_and_ite = chosen_configs_data[(chosen_configs_data["repetition"] == row["repetition"]) & (chosen_configs_data["iteration"] == row["iteration"])]
        if feature.is_interaction():
            option1, option2 = feature.get_options()
            return chosen_configs_data_same_rep_and_ite[(chosen_configs_data_same_rep_and_ite[option1] > 0) & (chosen_configs_data_same_rep_and_ite[option2] > 0)].shape[0] > 0
        else:
            return chosen_configs_data_same_rep_and_ite[chosen_configs_data_same_rep_and_ite[feature.get_option1()] > 0].shape[0] > 0

    def is_feature_selected(self, row, chosen_configs_data):
        feature = row["feature"]
        chosen_configs_data_same_rep_and_ite = chosen_configs_data[(chosen_configs_data["repetition"] == row["repetition"]) & (chosen_configs_data["iteration"] == row["iteration"])]
        return chosen_configs_data_same_rep_and_ite[chosen_configs_data_same_rep_and_ite["feature"] == feature].shape[0] > 0


    def plot_feature_selection_and_ci_width_change_per_function_and_feature(self, data: pd.DataFrame,
                                                                            chosen_configs_data: pd.DataFrame,
                                                                            model_names: list[str],
                                                                            feature_names: list[Feature],
                                                                            function_names: list[str] = ["system"]):
        """
        Plots a graph for each of the given feature showing the ci change with the info if the feature was selected and if it is "on" over the time of the active sampling process.
        Note: Numeric options can not really be assigned the value "on" or "off" if they do not have a 0 value.

        :param data: data to generate plot from.
        """
        for function_name in function_names:
            for model_name, is_active in self.get_model_and_active_zip(model_names):
                model_data = self.get_model_data_for_function(data, function_name, is_active, model_name)
                # TODO: model name is ignored here
                if model_data.shape[0] == 0:
                    continue

                ci_interval_width_change = self.calculate_ci_interval_width_change(model_data, model_names)
                ci_interval_width_change.loc[:, "is_selected"] = ci_interval_width_change.apply(lambda r: self.is_feature_selected(r, chosen_configs_data), axis=1)

                for feature in feature_names:
                    feature_ci_interval_width_change = ci_interval_width_change[ci_interval_width_change["feature"] == str(feature)]
                    if feature_ci_interval_width_change.shape[0] == 0:
                        continue

                    if feature_ci_interval_width_change["ci_interval_width_change"].isnull().all():
                        continue


                    feature_ci_interval_width_change = feature_ci_interval_width_change.sort_values(by=["iteration", "is_selected"], ascending=[True, False])
                    self.plot_scatter_plot(feature_ci_interval_width_change, "iteration", "ci_interval_width_change",
                                           "is_selected",
                                           f"ci interval width each repetition change for function {function_name} and feature {feature.as_pretty_string()}",
                                           "number of sampled configurations", "ci interval change",
                                           "ci_interval_width_change_over_active_sampling_for_function_and_feature/",
                                           f"ci_interval_width_change_over_active_sampling_for_function_{function_name}_and_feature_{feature.as_file_name_string()}.png",
                                           has_axhline=True,
                                           **{"axes": {"ylim": (-0.5, 0.5)}})



    def plot_ci_interval_width_change_of_top_uncertain_features_per_function(self, data: pd.DataFrame, model_names, n_top_uncertain_features: int, function_names: list[str] = ["system"]):
        ci_interval_width_change = self.calculate_ci_interval_width_change(data, model_names)

        results = pd.DataFrame(columns=ci_interval_width_change.columns)

        for model_name, is_active in self.get_model_and_active_zip(model_names):
            model_data = self.get_model_data(data, is_active, model_name)
            model_ci_interval_width_change = ci_interval_width_change.copy()

            if model_data.shape[0] == 0:
                continue

            model_ci_interval_width_change.loc[:, "model_name"] = model_name
            grouped_data = model_data.groupby(["repetition", "iteration"])
            model_ci_interval_width_change.loc[:, "is_top_feature"] = model_ci_interval_width_change.apply(
                lambda r: (r["repetition"], r["iteration"]) in grouped_data.groups.keys() and r["feature"] in list(grouped_data.get_group(
                    (r["repetition"], r["iteration"])).head(n_top_uncertain_features)["feature"]), axis=1)
            model_ci_interval_width_change = model_ci_interval_width_change[model_ci_interval_width_change["is_top_feature"] == True]

            results = pd.concat([results, model_ci_interval_width_change], ignore_index=True)

        for function_name in function_names:
            self.plot_line_graph(results[results["function_name"] == function_name], "iteration",
                                 "ci_interval_width_change", "model_name",
                                 f"mean ci interval width change of the top {n_top_uncertain_features} uncertain features for function {function_name}",
                                 "number of sampled configurations", "mean ci interval width change", "mean_ci_interval_width_change_for_function/",
                                 f"mean_ci_interval_width_change_for_function_{function_name}.png", "is_active")


    def get_ci_width_change(self, data, row):
        next_rep, next_ite = self.increment_repetition_and_iteration(data, row["repetition"], row["iteration"])
        fil = (data["function_name"] == row["function_name"]) & (data["feature"] == row["feature"]) & (data["repetition"] == next_rep) & (data["iteration"] == next_ite)
        if bool(fil.any()) is False:
            return np.nan

        return data[fil]["ci_interval_width"].iloc[0] - row["ci_interval_width"]

    def calculate_ci_interval_width_change(self, data, model_names):
        result = pd.DataFrame()
        for model_name, is_active in self.get_model_and_active_zip(model_names):
            model_data = self.get_model_data(data, is_active, model_name)

            if model_data.shape[0] == 0:
                continue

            model_data.loc[:, "ci_interval_width"] = abs(model_data["max_influence"] - model_data["min_influence"])
            model_data.loc[:, "ci_interval_width_change"] = np.nan
            model_data = model_data.sort_values(by=["repetition", "iteration", "ci_interval_width"], ascending=[True, True, False])

            model_data.loc[:, "ci_interval_width_change"] = model_data.apply(lambda row: self.get_ci_width_change(model_data, row), axis=1)
            model_data = model_data.drop(['influence', 'min_influence', 'max_influence', 'mape'], axis=1)

            result = pd.concat([result, model_data])

        result = self.sort_for_plotting(result)
        return result

    def plot_active_learning_selection_frequency_of_single_option_feature_values(self, chosen_configs_data: pd.DataFrame, features: list[Feature]) -> None:
        """
        Plots a bar chart that displays the total value count in percent of the feature and a line plot that displays the value count over the active learning process.
        This function handles feature with only a single option. For interacting features see plot_active_learning_selection_frequency_of_interaction_feature_values

        :param chosen_configs_data:
        :param features:
        """
        for feature in features:
            option = feature.get_option1()
            option_data = chosen_configs_data[option].value_counts(normalize=True).mul(100).round(1)
            option_data = option_data.reset_index().sort_values("index")
            option_data = option_data.rename(
                columns={"index": "value", option: "frequency"})

            self.plot_bar_plot(option_data, "value", "frequency", f"value selection frequency of option {option}",
                               "value", "value selection frequency in percent", "bar_plot_value_selection_frequency_of_option/",
                               f"bar_plot_value_selection_frequency_of_option_{option}")

            possible_values = list(chosen_configs_data[option].unique())
            repetitions = list(chosen_configs_data["repetition"].unique())
            result = pd.DataFrame()

            # split in reps to apply cumsum for each rep
            for repetition in repetitions:
                rep_data = chosen_configs_data[chosen_configs_data["repetition"] == repetition].groupby(["repetition", "iteration"])[option].value_counts().unstack().fillna(0).cumsum().reset_index().groupby(["repetition", "iteration"]).agg({value: "mean" for value in possible_values}).apply(lambda x: x/x.sum() * 100, axis=1).reset_index()
                result = pd.concat([result, rep_data])

            result = result.melt(id_vars=["repetition", "iteration"], var_name="value", value_name="frequency").sort_values(by=["repetition", "iteration", "value"])

            self.plot_line_graph(result, "iteration", "frequency", "value",
                                 f"value selection frequency for option {option}", "number of sampled configurations",
                                 "value selection frequency in percent", "value_selection_frequency_of_option/",
                                 f"value_selection_frequency_of_option_{option}")

    def plot_active_learning_selection_frequency_of_interaction_feature_values(self, chosen_configs_data, features: list[Feature]):
        """
        Plots a bar chart that displays the total value count in percent of the feature and a line plot that displays the value count over the active learning process.
        This function handles only features with interacting options. For single option features see plot_active_learning_selection_frequency_of_single_feature_values

        :param chosen_configs_data:
        :param features:
        """
        for feature in features:
            option1, option2 = feature.get_options()
            option_data = chosen_configs_data[[option1, option2]]
            option_data = option_data.value_counts(normalize=True).mul(100).round(1)
            option_data = option_data.sort_index().reset_index()
            option_data = option_data.rename(columns={0: "frequency"})
            option_data.loc[:, "value"] = "(" + option_data[option1].astype(str) + ", " + \
                                           option_data[
                                               option2].astype(str) + ")"

            self.plot_bar_plot(option_data, "value", "frequency",
                               f"value selection frequency of option {feature.as_pretty_string()}", "value",
                               "value selection frequency in percent", "bar_plot_value_selection_frequency_of_feature/",
                               f"bar_plot_value_selection_frequency_of_feature_{feature.as_file_name_string()}")
            possible_values = [tuple(el) for el in
                               list(np.unique(chosen_configs_data[[option1, option2]].values, axis=0))]
            repetitions = list(chosen_configs_data["repetition"].unique())
            result = pd.DataFrame(columns=["repetition", "iteration"] + [str(el) for el in
                                                                         possible_values])  # due to super strange multi index problems
            # split in reps to apply cumsum for each rep
            for repetition in repetitions:
                rep_data = chosen_configs_data[chosen_configs_data["repetition"] == repetition].groupby(
                    ["repetition", "iteration"])[[option1, option2]].value_counts().unstack([-2, -1]).fillna(
                    0).cumsum().apply(lambda x: x / x.sum() * 100, axis=1).reset_index()

                rep_result = pd.DataFrame(columns=["repetition", "iteration"] + [str(el) for el in possible_values])
                for column in list(rep_result.columns):  # due to super strange multi index problems
                    eval_column = eval(column) if "," in column else column
                    if eval_column in rep_data.columns:
                        rep_result[column] = list(rep_data[eval_column])
                    else:
                        rep_result[column] = np.nan

                result = pd.concat([result, rep_result], ignore_index=True)
            result = result.melt(id_vars=["repetition", "iteration"], value_vars=[str(el) for el in possible_values],
                                 var_name="value",
                                 value_name="frequency").sort_values(by=["repetition", "iteration", "value"])
            self.plot_line_graph(result, "iteration", "frequency", "value",
                                 f"value selection frequency for option {feature.as_pretty_string()}",
                                 "number of sampled configurations", "value selection frequency in percent",
                                 "value_selection_frequency_of_feature/", f"value_selection_frequency_of_feature_{feature.as_file_name_string()}")


    def plot_features_vif(self, feature_data: pd.DataFrame, features: list[Feature]) -> None:
        """
        Plots a subplot for each feature that displays the variance inflation factor (VIF) of the feature over the active learning process.
        """
        for feature in features:
            data = feature_data[feature_data["feature"] == str(feature)]
            self.plot_line_graph(data, "iteration", "vif", None,
                                 f"VIF for feature {feature.as_pretty_string()}", "number of sampled configurations",
                                 "VIF", "feature_vif/",
                                 f"feature_vif_for_feature_{feature.as_file_name_string()}")

    def get_repetitions_and_iterations(self, data):
        """
        Gets all standard repetitions and iterations for given dataframe.

        :param data: frame to retrieve reps and ites from.
        :return: zipped list of all rep and ite tuples.
        """
        iterations = [list(set(iterations)) for iterations in
                      data.groupby(["repetition"])["iteration"].apply(list)]

        estimated_batch_size = self.get_estimated_batch_size(iterations)
        standard_iterations = self.get_standard_iterations(estimated_batch_size, iterations)
        n_standard_iterations = len(standard_iterations)

        repetition_range = list(range(data["repetition"].min(), data["repetition"].max()))
        repetition_range = [data["repetition"].min()] if not repetition_range else repetition_range
        n_repetition_range = len(repetition_range)
        reps_and_ites = [(rep, ite) for rep, ite in zip(sorted(repetition_range * n_standard_iterations), standard_iterations * n_repetition_range)]
        return reps_and_ites


    def increment_repetition_and_iteration(self, data: pd.DataFrame, rep: int, ite: int):
        max_rep = data["repetition"].max()
        max_ite = data["iteration"].max()
        min_ite = data["iteration"].min()
        iterations = [iterations for iterations in
                      data.groupby(["repetition"])["iteration"].unique().apply(list)]
        batch_size = self.get_estimated_batch_size(iterations)

        if batch_size is None:
            return None, None

        if ite < max_ite:
            ite += batch_size
        else:
            if rep < max_rep:
                rep += 1
                ite = min_ite
            else:
                return None, None

        return rep, ite

    def evaluate_filter(self, data_file_name: str):
        data = self.data_reader.read_in_data(data_file_name)
        function_names = list(data["method"].unique())
        total_runtime = self.data_handler.get_total_runtime(data)
        print(f"total runtime is: {total_runtime}")
        top_functions_with_runtime = {}
        sum_other_functions_runtime = 0
        smallest_runtime = 1000
        smallest_function = ""

        for function_name in function_names:
            function_data = data[data["method"] == function_name]
            function_runtime = np.sum(function_data["time"])

            if function_runtime < smallest_runtime:
                smallest_runtime = function_runtime
                smallest_function = function_name

            if function_runtime / total_runtime > self.runtime_threshold:
                top_functions_with_runtime[function_name] = {"runtime": int(function_runtime), "runtime_fraction": round(100 * function_runtime / total_runtime, 2)}
            else:
                sum_other_functions_runtime += function_runtime

        print(sorted(top_functions_with_runtime.items(), key=lambda x: x[1]["runtime"], reverse=True))
        sum_other_functions_runtime_fraction = round(100 * sum_other_functions_runtime / total_runtime, 2)
        print(f"other functions runtime sum is {int(sum_other_functions_runtime)} and their fraction on total runtime is {sum_other_functions_runtime_fraction}")
        print(f"function with shortest runtime is {smallest_function} with runtime {smallest_runtime}")


    def plot_features_vif_with_missing_values(self, data_file_name: str, features, features_vif_data, features_vif_data_with_first_missing_value, features_vif_data_with_second_missing_value, vif_infos: list[str]):
        """
        Plots vif values for given features in subplots comparing experiments with untouched numeric feature values and those
        where values were removed from numeric features before conducting the experiment.

        :param features:
        :return:
        """
        missing_option, first_missing_value, second_missing_value = vif_infos
        data = self.data_reader.read_in_data(data_file_name)
        missing_feature = Feature(feature_as_tuple=(missing_option,))
        numeric_option_names = [missing_feature] + [feature for feature in features if len(data[feature.get_option1()].unique()) > 2 if feature != missing_feature][:2] # missing feature + first two numeric features

        features_vif_data.loc[:, "missing_value"] = "None"
        features_vif_data_with_first_missing_value.loc[:, "missing_value"] = str(first_missing_value)
        features_vif_data_with_second_missing_value.loc[:, "missing_value"] = str(second_missing_value)

        #self.plot_vif_graph(features, features_vif_data, title="Variance Inflation Factor (VIF) for numeric features", file_name=f"features_vif_for_feature_{'_and_'.join([feature.as_file_name_string() for feature in features])}")

        features_vif = pd.concat([features_vif_data, features_vif_data_with_first_missing_value, features_vif_data_with_second_missing_value], ignore_index=True)
        self.plot_vif_graph(numeric_option_names, features_vif, title="Variance Inflation Factor (VIF) for different missing values in numeric features", file_name=f"features_vif_with_missing_values_for_feature_{'_and_'.join([feature.as_file_name_string() for feature in numeric_option_names])}")

    def plot_vif_graph(self, features, features_vif, title, file_name):
        plt.clf()
        fig, axs = plt.subplots(ncols=len(features), figsize=(11, 7))
        fig.subplots_adjust(wspace=1)
        for feature in features:
            sns.lineplot(data=features_vif[features_vif["feature"] == str(feature)], x="iteration", y="vif",
                         hue="missing_value", ax=axs[features.index(feature)])
            current_axis = axs[features.index(feature)]
            current_axis.set_title(feature.as_pretty_string())
            current_axis.set_xlabel("number of sampled configurations")
            current_axis.set(ylim=(0, 20))
            current_axis.legend(loc="upper right")
        # keep legend only for last plot
        for i in range(len(features) - 1):
            axs[features.index(features[i])].get_legend().remove()
        fig.suptitle(title, fontsize=16)
        self.data_reader.create_images_directory_if_not_exists("features_vif")
        plt.savefig(
            f"{self.path_to_images}features_vif/{file_name}",
            bbox_inches="tight")

    def plot_correlation_n_configs_p4_failure(self, data_file_name: str, function_names, results_data: pd.DataFrame, model_names: list[str]):
        data = self.data_reader.read_in_data(data_file_name)
        total_runtime = self.data_handler.get_total_runtime(data)
        functions_with_n_configs = {}

        for function_name in function_names:
            function_data = data[data["method"] == function_name]
            n_configs = data[data["method"] == function_name].shape[0]
            function_runtime = np.sum(function_data["time"])
            functions_with_n_configs[function_name] = {"n_configs": n_configs, "runtime_fraction": round(100 * function_runtime / total_runtime, 2)}

        for model_name, is_active in self.get_model_and_active_zip(model_names):
            model_data = self.get_model_data(results_data, is_active, model_name)

            if model_data.shape[0] == 0:
                continue

            if len(list(model_data["function_name"].unique())) == 1 and list(model_data["function_name"].unique())[0] == "system":
                return

            n_iterations = len(model_data["iteration"].unique())
            n_reps = 10

            ites_per_feat = model_data.groupby(["function_name", "repetition", "iteration"]).agg(
                {"influence": "mean"}).reset_index()
            ites_per_feat = ites_per_feat.groupby(["function_name"]).agg({"iteration": "count"}).reset_index()
            ites_per_feat["fraction"] = 100 * (1 - (ites_per_feat["iteration"] / (n_reps * n_iterations)))
            ites_per_feat["n_configs"] = ites_per_feat["function_name"].apply(lambda x: functions_with_n_configs[x]["n_configs"])
            ites_per_feat["runtime_fraction"] = ites_per_feat["function_name"].apply(lambda x: functions_with_n_configs[x]["runtime_fraction"])

            self.plot_scatter_plot(ites_per_feat, "n_configs", "fraction", "runtime_fraction",
                                   "Correlation between number of measured configurations, runtime fraction and frequency of P4 failure",
                                   "number of measured configurations", "failure frequency in %",
                                   f"correlation_n_configs_runtime_p4_failure/",
                                   f"correlation_n_configs_runtime_p4_failure.png", **{"axes": {"ylim": (0, 100)}})


    def plot_correlation_runtime_p4_failure(self, data_file_name: str, results_data: pd.DataFrame, model_names: list[str]):
        data = self.data_reader.read_in_data(data_file_name)
        function_names = list(data["method"].unique())
        total_runtime = self.data_handler.get_total_runtime(data)
        top_functions_with_runtime = {}

        for function_name in function_names:
            function_data = data[data["method"] == function_name]
            function_runtime = np.sum(function_data["time"])

            if function_runtime / total_runtime > self.runtime_threshold:
                top_functions_with_runtime[function_name] = {"runtime": int(function_runtime), "runtime_fraction": round(100 * function_runtime / total_runtime, 2)}

        for model_name, is_active in self.get_model_and_active_zip(model_names):
            model_data = self.get_model_data(results_data, is_active, model_name)

            if model_data.shape[0] == 0:
                continue

            if len(list(model_data["function_name"].unique())) == 1 and list(model_data["function_name"].unique())[0] == "system":
                return

            n_iterations = len(model_data["iteration"].unique())
            n_reps = 10

            ites_per_feat = model_data.groupby(["function_name", "repetition", "iteration"]).agg(
                {"influence": "mean"}).reset_index()
            ites_per_feat = ites_per_feat.groupby(["function_name"]).agg({"iteration": "count"}).reset_index()
            ites_per_feat["fraction"] = 1 - (100 * ites_per_feat["iteration"] / (n_reps * n_iterations))
            ites_per_feat["runtime_fraction"] = ites_per_feat["function_name"].apply(lambda x: top_functions_with_runtime[x]["runtime_fraction"])

            self.plot_scatter_plot(ites_per_feat, "runtime_fraction", "fraction", None,
                                   "Correlation between runtime fraction and frequency of P4 failure",
                                   "runtime fraction in %", "failure frequency in %",
                                   f"correlation_runtime_p4_failure/", f"correlation_runtime_p4_failure.png", **{"axes": {"ylim": (0, 100)}})

    def evaluate_data(self, data_frames,
                      model_names, path_modifier: str, dataIsCombined=False, function_names: list[str] = ["system"],
                      system_name=None, vif_infos = None):

        results_data, chosen_configs_data, features_vif_data, features_vif_data_with_first_missing_value, features_vif_data_with_second_missing_value = data_frames

        original_path = self.path_to_images
        self.path_to_images += path_modifier
        self.data_reader.path_to_images = self.path_to_images
        p4_models = [model_name for model_name in model_names if "p4" in model_name]
        all_features = [Feature(feature_as_string_comma_seperated=feature) for feature in list(results_data["feature"].unique())]
        single_option_features = [feature for feature in all_features if feature.is_interaction() is False]
        interaction_features = [feature for feature in all_features if feature.is_interaction() is True]

        if vif_infos is not None:
            self.plot_features_vif_with_missing_values(f"{system_name}_ml_cfg", single_option_features,
                                                       features_vif_data, features_vif_data_with_first_missing_value,
                                                       features_vif_data_with_second_missing_value, vif_infos)


        self.evaluate_filter(f"{system_name}_ml_cfg")
        self.plot_features_vif(features_vif_data, all_features)
        self.plot_correlation_runtime_p4_failure(f"{system_name}_ml_cfg", results_data, p4_models)
        self.plot_correlation_n_configs_p4_failure(f"{system_name}_ml_cfg", function_names, results_data, p4_models)
        self.plot_influence_of_all_features_and_individual_top_features_for_repetitions(results_data, p4_models, 5)
        self.plot_active_learning_selection_frequency_of_single_option_feature_values(chosen_configs_data, single_option_features)
        self.plot_active_learning_selection_frequency_of_interaction_feature_values(chosen_configs_data, interaction_features)
        self.plot_influence_of_all_features_and_average_top_features(results_data, p4_models, 5)
        self.plot_ci_interval_width_of_all_features_and_top_features_over_active_learning_process(results_data, p4_models, 5)
        self.plot_ci_interval_width_of_all_features_and_individual_top_features_for_repetition(results_data, p4_models, 5)
        self.plot_feature_selection_and_ci_width_change_per_function_and_feature(results_data, chosen_configs_data, p4_models, all_features)
        self.plot_ci_interval_width_change_of_top_uncertain_features_per_function(results_data, p4_models, 5)
        self.plot_mape_for_function(results_data)
        self.plot_feature_selection_count_and_influence_over_active_learning_process(results_data, model_names)
        self.plot_influence_of_feature_selection_over_active_learning_process(results_data, model_names)

        if dataIsCombined is False:
            self.plot_model_feature_selection_and_their_influences(results_data, [model_name for model_name in model_names if "lasso" in model_name or "p4" in model_name])

        if len(function_names) > 1:
            self.plot_influence_of_all_features_and_individual_top_features_for_repetitions(results_data, p4_models, 5, function_names)
            self.plot_influence_of_all_features_and_average_top_features(results_data, p4_models, 5, function_names)
            self.plot_ci_interval_width_of_all_features_and_top_features_over_active_learning_process(results_data, p4_models, 5, function_names)
            self.plot_ci_interval_width_of_all_features_and_individual_top_features_for_repetition(results_data, p4_models, 5, function_names)
            self.plot_feature_selection_and_ci_width_change_per_function_and_feature(results_data, chosen_configs_data, p4_models, all_features, function_names)
            self.plot_ci_interval_width_change_of_top_uncertain_features_per_function(results_data, p4_models, 5, function_names)
            self.plot_mape_for_function(results_data, function_names)
            self.plot_feature_selection_count_and_influence_over_active_learning_process(results_data, model_names, function_names)
            self.plot_influence_of_feature_selection_over_active_learning_process(results_data, model_names, function_names)

            if dataIsCombined is False:
                self.plot_model_feature_selection_and_their_influences(results_data, p4_models, function_names)


        # set path back to original path
        self.path_to_images = original_path

def main():
    is_debug = False

    parser = argparse.ArgumentParser(description="Evaluating results from pipeline.")
    parser.add_argument("--system-name", help="System to be evaluated. E.g. x264", type=str)
    parser.add_argument("--base-file", help="Base file name to read results from.", type=str)
    parser.add_argument("--base-directory", help="Base directory to save images to.", type=str)
    parser.add_argument("--missing-option", help="Option that is missing values.", type=str)
    parser.add_argument("--first-missing-value", help="Value that is missing for option.", type=str)
    parser.add_argument("--second-missing-value", help="Value that is missing for option.", type=str)
    args = parser.parse_args()

    args = get_arguments() if is_debug else args

    evaluation = ActiveBayesifyEvaluation(args.system_name)
    model_names = ["p4", "lasso"]

    results_file_name = f"{args.base_file}_results"
    chosen_configs_file_name = f"{args.base_file}_chosen_configs"
    features_vif_file_name = f"{args.base_file}_features_vif"

    path = f"{args.base_directory}"

    data = evaluation.data_reader.read_in_results(results_file_name)
    chosen_configs_data = evaluation.data_reader.read_in_results(chosen_configs_file_name)
    features_vif_data = evaluation.data_reader.read_in_results(features_vif_file_name)
    features_vif_data_with_first_missing_value, features_vif_data_with_second_missing_value = None, None
    vif_infos = None

    if args.missing_option is not None:
        features_vif_with_first_missing_value = f"{args.base_file}_with_modified_dataset_missing_{args.missing_option}_{args.first_missing_value}_value_features_vif"
        features_vif_with_second_missing_value = f"{args.base_file}_with_modified_dataset_missing_{args.missing_option}_{args.second_missing_value}_value_features_vif"
        features_vif_data_with_first_missing_value = evaluation.data_reader.read_in_results(features_vif_with_first_missing_value)
        features_vif_data_with_second_missing_value = evaluation.data_reader.read_in_results(features_vif_with_second_missing_value)
        vif_infos = (args.missing_option, args.first_missing_value, args.second_missing_value)

    data_frames = (data, chosen_configs_data, features_vif_data, features_vif_data_with_first_missing_value, features_vif_data_with_second_missing_value)


    function_names = list(data["function_name"].unique())
    #function_names = ["get_ref"]

    evaluation.evaluate_data(data_frames, model_names, path, False, function_names, args.system_name, vif_infos)

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluating results from pipeline.")
    args = parser.parse_args()
    args.system_name = "lrzip"
    args.base_file = "active_learning_on_function_level_with_weights"
    args.base_directory = "function_level/active_learning_with_weights/"
    args.missing_option = "ram"
    args.first_missing_value = "2"
    args.second_missing_value = "6"

    return args

if __name__ == "__main__":
    main()