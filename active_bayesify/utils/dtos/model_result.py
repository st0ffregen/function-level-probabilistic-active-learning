from active_bayesify.utils.models.feature import Feature


class ModelResult():

    def __init__(self, model_name: str, repetition: int, iteration: int, feature: Feature, mape: float,
                 influence: float, min_influence: float = None, max_influence: float = None,
                 function_name: str = "system"):
        self.model_name = model_name
        self.repetition = repetition
        self.iteration = iteration
        self.feature = feature
        self.mape = mape
        self.influence = influence
        self.min_influence = min_influence
        self.max_influence = max_influence
        self.function_name = function_name

    def as_dict(self):
        return {
            "model_name": self.model_name,
            "function_name": self.function_name,
            "repetition": self.repetition,
            "iteration": self.iteration,
            "feature": str(self.feature),
            "mape": self.mape,
            "influence": self.influence,
            "min_influence": self.min_influence,
            "max_influence": self.max_influence
        }

    def __eq__(self, other):
        if not isinstance(other, ModelResult):
            return NotImplemented

        return self.model_name == other.model_name and self.repetition == other.repetition and \
            self.iteration == other.iteration and self.feature == other.feature and \
            self.mape == other.mape and self.influence == other.influence and self.min_influence == other.min_influence and \
            self.max_influence == other.max_influence and self.function_name == other.function_name
