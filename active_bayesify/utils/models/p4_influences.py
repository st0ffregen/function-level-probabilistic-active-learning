import numpy as np

from active_bayesify.utils.models.feature import Feature


class P4Influences:

    def __init__(self, function_name: str, p4_influences_as_dict: dict = None, p4_influences_as_feature_list: list[Feature] = None):
        self.function_name = function_name
        if p4_influences_as_dict:
            self.influences = self.parse_p4_dict_to_feature_list(p4_influences_as_dict)
        elif p4_influences_as_feature_list or p4_influences_as_feature_list == []:
            self.influences = p4_influences_as_feature_list
        else:
            raise Exception("Can not instantiate P4Influences object with these arguments!")

        self.influences = self.merge_feature_with_same_options()

    def parse_p4_dict_to_feature_list(self, p4_influences: dict) -> list[Feature]:
        return [Feature(feature_as_tuple=key, influence=value) for key, value in p4_influences.items()]

    def merge_feature_with_same_options(self) -> list[Feature]:
        """
        Merge two features if they have same options. Used because unclear if P4 returns dict without duplicates.
        """
        new_influences = []
        for feature in self.influences:
            feature_with_only_option_names = Feature(option1=feature.get_option1(), option2=feature.get_option2())
            new_influences_with_only_option_names = [Feature(option1=feature.get_option1(), option2=feature.get_option2()) for feature in new_influences]
            if feature_with_only_option_names not in new_influences_with_only_option_names:
                new_influences.append(feature)
            else:
                idx = new_influences_with_only_option_names.index(feature_with_only_option_names)

                if new_influences[idx].influence_is_array() and feature.influence_is_array():
                    new_influences[idx].set_influence(np.mean([new_influences[idx].get_influence(), feature.get_influence()], axis=0))
                elif new_influences[idx].influence_is_array() is False and feature.influence_is_array() is False:
                    new_influences[idx].set_influence(np.mean([new_influences[idx].get_influence(), feature.get_influence()]))
                else:
                    raise Exception("Can not calculate mean between numpy array and integer!")

        return new_influences

    def get_features(self) -> list[Feature]:
        return self.influences

    def set_features(self, p4_influences: dict):
        self.parse_p4_dict_to_feature_list(p4_influences)
        self.merge_feature_with_same_options()

    def get_features_sorted_by_uncertainty(self, desc: bool = True) -> list[Feature]:
        if self.influences == []:
            return []
        if all([isinstance(feature.get_influence(), np.ndarray) is False for feature in self.influences]):
            return NotImplemented
        return sorted(self.influences, key=lambda x: abs(x.get_influence()[0] - x.get_influence()[1]), reverse=desc)

    def get_function_name(self) -> str:
        return self.function_name

    def __eq__(self, other):
        return self.function_name == other.function_name and self.influences == other.influences
