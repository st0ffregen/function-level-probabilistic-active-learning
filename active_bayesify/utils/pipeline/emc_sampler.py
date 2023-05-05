from logging import Logger

import numpy as np
import pandas as pd
import multiprocessing as mp

from active_bayesify.utils.models.p4_influences import P4Influences
from active_bayesify.utils.pipeline.data_handler import DataHandler


class EMCSampler:

    def __init__(self, data_handler: DataHandler, logger: Logger, option_names: list[str], train_function: callable,
                 y_column_names: list[str] = ["taskID", "method", "energy"]):
        self.option_names = option_names
        self.data_handler = data_handler
        self.logger = logger
        self.train_function = train_function
        self.option_names_with_id = ["taskID"] + option_names
        self.option_names_with_id_and_method = self.option_names_with_id + ["method"]
        self.y_column_names = y_column_names

    def actively_sample_new_instances_with_parameter_change(self, X_pool, y_pool, X_train, y_train, X_test, y_test,
                                                            p4_influences: P4Influences, p4_models, p4_preprocessors,
                                                            function_to_run_emc_for: str, random_state: int,
                                                            batch_size: int = 5, is_real_emc: bool = True,
                                                            is_multiprocessing=True, key_column: str = "taskID"):
        """
        Active sampling of new instances based on highest mean uncertainty reduction calculated over all features.
        :param function_to_run_emc_for:
        :param is_multiprocessing:
        """
        if len(p4_influences.get_features()) == 0:
            self.logger.info("sample randomly because no features are given.")
            return self.data_handler.randomly_sample_new_instances(X_pool, y_pool, batch_size, random_state)

        p4_influences_list_expected = {}

        # TODO: this implementation makes decision based on best instances for the first function in the list

        X_pool_summed = X_pool[X_pool["method"] == function_to_run_emc_for]
        X_train_summed = X_train[X_train["method"] == function_to_run_emc_for]
        X_pool_summed = X_pool_summed.reset_index(drop=True) # for correct idx assignment in for loop

        prior_influence_mean_width = np.mean([abs(feature.get_influence()[0] - feature.get_influence()[1]) for feature in p4_influences.get_features()])

        args_list = [(X_pool, X_pool_summed, X_test, X_train_summed, idx, p4_influences,
                                  p4_influences_list_expected, p4_models, p4_preprocessors, random_state, row, y_pool,
                                  y_test, y_train, is_real_emc) for idx, row in X_pool_summed.iterrows()]

        if is_multiprocessing:
            with mp.get_context("spawn").Pool(processes=min((mp.cpu_count() - 1), 20)) as pool:
                pool_results = pool.map(self.train_with_new_instance, args_list)

            for pool_result in pool_results:
                taskId, p4_influences = pool_result
                if p4_influences is None:
                    continue
                p4_influences_list_expected[taskId] = p4_influences
        else:
            for args in args_list:
                taskId, p4_influences = self.train_with_new_instance(args)
                if p4_influences is None:
                    continue
                p4_influences_list_expected[taskId] = p4_influences

        p4_influences_list_expected = dict(sorted(p4_influences_list_expected.items(), key=lambda x: self.get_mean_difference(x[1], prior_influence_mean_width), reverse=True))

        task_ids_with_best_uncertainty_reduction = list(p4_influences_list_expected.keys())[:batch_size]

        if len(task_ids_with_best_uncertainty_reduction) < batch_size:
            remaining_instances_to_sample = batch_size - len(task_ids_with_best_uncertainty_reduction)
            self.logger.info(f"Randomly sample {remaining_instances_to_sample} because active sampling did not return enough instances.")
            task_ids_with_best_uncertainty_reduction += self.data_handler.randomly_sample_new_instances(X_pool[~X_pool["taskID"].isin(task_ids_with_best_uncertainty_reduction)], y_pool, remaining_instances_to_sample, random_state)

        return X_pool[X_pool["taskID"].isin(task_ids_with_best_uncertainty_reduction)], y_pool[y_pool["taskID"].isin(task_ids_with_best_uncertainty_reduction)]

    def train_with_new_instance(self, args):
        X_pool, X_pool_summed, X_test, X_train_summed, idx, p4_influences, p4_influences_list_expected, p4_models, p4_preprocessors, random_state, row, y_pool, y_test, y_train, is_real_emc = args
        self.logger.info(
            f"Train on new instance with taskId {row['taskID']}. {X_pool_summed.shape[0] - idx} instances left to test.")
        # TODO: this implementation makes decision based on best instances for the first function in the list
        function_to_run_emc_for = list(X_pool["method"])[0]
        y_row = y_pool[(y_pool["method"] == function_to_run_emc_for) & (y_pool["taskID"] == row["taskID"])]
        X_row_as_frame = row.to_frame().T
        X_row_as_frame = X_row_as_frame.astype({column: X_pool[column].dtype for column in X_pool.columns})
        X_train_new = pd.concat([X_train_summed, X_row_as_frame], ignore_index=True)
        X_train_new = X_train_new.astype({column: X_pool[column].dtype for column in X_pool.columns})

        if is_real_emc:
            if function_to_run_emc_for not in p4_preprocessors:
                return None, None

            X_row_numpy = X_row_as_frame[self.option_names].to_numpy()
            X_row_transformed = p4_preprocessors[function_to_run_emc_for].transform(X_row_numpy)
            prediction_with_n_samples = p4_models[function_to_run_emc_for].predict(X_row_transformed, n_samples=1000)
            prediction = float(np.mean(prediction_with_n_samples, axis=0)[0])
            y_row["energy"] = prediction
        y_train_new = pd.concat([y_train[(y_train["method"] == function_to_run_emc_for)], y_row], ignore_index=True)
        p4_influences, _, _, _ = self.train_function(function_to_run_emc_for, 0, idx, X_train_new, X_test, y_train_new, y_test,
                                                     random_state)
        return row["taskID"], p4_influences

    def get_mean_difference(self, p4_influences: P4Influences, prior_influence_mean_width: float):
        influence_mean_width = np.mean([abs(feature.get_influence()[0] - feature.get_influence()[1]) for feature in p4_influences.get_features()])
        return prior_influence_mean_width - influence_mean_width
