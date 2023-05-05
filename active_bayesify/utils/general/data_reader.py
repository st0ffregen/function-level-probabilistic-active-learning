import pandas as pd
import os
from active_bayesify.utils.general.config_parser import ConfigParser


class DataReader:

    def __init__(self, config_parser: ConfigParser):
        """
        Initializes `DataReader` instance. Provides function to read in functions and their data.

        :param config_parser: To get paths to data and results.
        """
        self.path_to_data = config_parser.get_path_with_system_name("Data")
        self.path_to_results = config_parser.get_path_with_system_name("Results")
        self.path_to_images = config_parser.get_path_with_system_name("Images")

    def get_all_functions(self) -> list:
        """
        Reads in all function names from `function_names.csv` file.

        :return: string list of function names.
        """
        return pd.read_csv(f"{self.path_to_data}function_names.csv")["method"]

    def get_functions_with_data(self) -> list:
        """
        Scans result directory for already measured functions.

        :return: string list of already measured function names.
        """
        files = []

        with os.scandir(self.path_to_results) as directory:
            for entry in directory:
                if entry.name[-3:] == "csv":
                    files.append(entry.name[:-6])  # removes file ending and repetition run information

        return list(set(files))

    def read_in_data(self, file_name: str) -> pd.DataFrame:
        """
        Reads in data from a file in the data directory.

        :param file_name: the name of the file.
        :return: pandoc data frame containing the data.
        """
        return pd.read_csv(f"{self.path_to_data}{file_name}.csv")

    def read_in_results(self, file_name: str) -> pd.DataFrame:
        """
        Reads in data from a results file in the results directory.

        :param file_name: file name where results are saved.
        :return: pandoc data frame with results from model.
        """
        return pd.read_csv(f"{self.path_to_results}{file_name}.csv")

    # TODO: this is not a read function!
    def create_directory_if_not_exists(self, path, directory_name=None):
        if directory_name is None:
            directory_name = ""

        if not os.path.exists(f"{path}{directory_name}"):
            os.makedirs(f"{path}{directory_name}")

    def create_images_directory_if_not_exists(self, directory_name):
        self.create_directory_if_not_exists(self.path_to_images, directory_name)