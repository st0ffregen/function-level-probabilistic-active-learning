import logging
from datetime import datetime
from active_bayesify.utils.general.config_parser import ConfigParser


class Logger:

    def __init__(self, config_parser: ConfigParser):
        """
        Provides a logger to be accessed from logger variable.

        :param config_parser: To get logs path.
        """
        path_to_logs = config_parser.get("Paths", "Logs")
        file_name = config_parser.get("Logging", "FileName")
        log_level = config_parser.get("Logging", "LogLevel")

        self.logger = logging.getLogger(__name__)

        if log_level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif log_level == "WARN":
            self.logger.setLevel(logging.WARN)

        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        file_handler = logging.FileHandler(f'{path_to_logs}{file_name}_{str(int(datetime.utcnow().timestamp()))}.log')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Makes logger accessible.

        :return: logger instance.
        """
        return self.logger
