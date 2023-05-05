import configparser


class ConfigParser(configparser.ConfigParser):

    def __init__(self, system_name: str):
        super().__init__()
        self.read("config.ini")
        self.system_name = system_name

    def get_path_with_system_name(self, option: str) -> str:
        path = self.get("Paths", option)
        modified_path = "/".join(path.split("/")[:-1]) + "/" + self.system_name + path.split("/")[-1] + "/"
        return modified_path
