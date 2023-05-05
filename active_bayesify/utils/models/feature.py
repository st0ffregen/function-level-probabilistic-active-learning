from typing import Union

import numpy as np


class Feature:

    option1 = ""
    option2 = None

    option1_value = None
    option2_value = None

    influence = None

    def __init__(
            self,
            feature_as_tuple: tuple[str, str] = None,
            feature_as_string_comma_seperated: str = None,
            feature_as_string_space_seperated: str = None,
            option1: str = None,
            option2: str = None,
            option1_value: int = None,
            option2_value: int = None,
            influence: Union[np.ndarray, float] = None
    ):
        if option1:
            self.option1 = option1
            if option2:
                self.option2 = option2
        elif feature_as_tuple:
            self.option1 = feature_as_tuple[0]
            self.option2 = feature_as_tuple[1] if len(feature_as_tuple) == 2 else None
        elif feature_as_string_comma_seperated:
            feature = eval(feature_as_string_comma_seperated)
            self.option1 = feature[0]
            self.option2 = feature[1] if len(feature) == 2 else None
        elif feature_as_string_space_seperated:
            split = feature_as_string_space_seperated.split(" ")
            self.option1, self.option2 = (split[0], split[1]) if len(split) == 2 else (split[0], None)
        else:
            raise Exception("Can not instantiate Feature object with these arguments!")

        if option1_value:
            self.option1_value = option1_value
            if option2_value:
                self.option2_value = option2_value
            else:
                raise Exception("Can not instantiate Feature object with these arguments!")

        if influence is not None:
            if isinstance(influence, np.ndarray):
                influence = influence.astype(float)
            else:
                influence = float(influence)
            self.influence = influence

        if option1_value and option2_value:
            self.sort_options_alphabetically_with_values()
        else:
            self.sort_options_alphabetically()

    def is_interaction(self) -> bool:
        return self.option2 is not None

    def influence_is_array(self) -> bool:
        return isinstance(self.influence, np.ndarray)

    def sort_options_alphabetically(self) -> None:
        if self.option2:
            self.option1, self.option2 = tuple(sorted((self.option1, self.option2)))

    def sort_options_alphabetically_with_values(self) -> None:
        if self.option2:
            real_option1, real_option2 = tuple(sorted((self.option1, self.option2)))

            if real_option1 != self.option1:
                real_option1_value = self.option2_value
                real_option2_value = self.option1_value
                self.option1_value, self.option2_value = real_option1_value, real_option2_value

            self.option1, self.option2 = real_option1, real_option2

    def get_options(self) -> tuple:
        return self.option1, self.option2

    def get_options_as_list(self) -> list:
        return [self.option1, self.option2]

    def get_option1(self) -> str:
        return self.option1

    def get_option2(self) -> str:
        return self.option2

    def get_influence(self) -> Union[np.ndarray, float]:
        return self.influence

    def set_option1(self, option1: str) -> None:
        self.option1 = option1
        self.sort_options_alphabetically()

    def set_option2(self, option2: str) -> None:
        self.option2 = option2
        self.sort_options_alphabetically()

    def set_option1_value(self, option1_value: int) -> None:
        self.option1_value = option1_value

    def set_option2_value(self, option2_value: int) -> None:
        self.option2_value = option2_value

    def set_influence(self, influence: Union[np.ndarray, float]) -> None:
        self.influence = influence

    def __str__(self):
        return str((self.option1, self.option2)) if self.option2 else str((self.option1,))

    def as_pretty_string(self) -> str:
        if self.is_interaction():
            return "(" + self.option1 + ", " + self.option2 + ")"

        return "(" + self.option1 + ")"

    def as_file_name_string(self) -> str:
        if self.is_interaction():
            return self.option1 + "_" + self.option2

        return self.option1

    def __eq__(self, other):
        if isinstance(self.influence, float):
            influences_are_same = self.influence == other.influence
        elif isinstance(self.influence, np.ndarray):
            influences_are_same = (self.influence == other.influence).all()
        elif self.influence is None:
            influences_are_same = True
        else:
            raise Exception("Can not compare Feature instances. Influences datatypes diverge!")

        return self.option1 == other.option1 and self.option2 == other.option2 and self.option1_value == self.option1_value and self.option2_value == other.option2_value and influences_are_same