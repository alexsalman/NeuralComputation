# https://github.com/rgeirhos/texture-vs-shape/blob/master/code/probabilities_to_decision.py
import numpy as np
from abc import ABC, abstractmethod
print('test')

import helper.human_categories as hc

class ProbabilitiesToDecisionMapping(ABC):
    @abstractmethod
    def probabilities_to_decision(self, probabilities):
        pass

    def check_input(self, probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

class ImageNetProbabilitiesTo16ClassesMapping(ProbabilitiesToDecisionMapping):
    def __init__(self, aggregation_function=np.mean):

        self.aggregation_function = aggregation_function

    def probabilities_to_decision(self, probabilities):
        self.check_input(probabilities)
        assert len(probabilities) == 1000

        max_value = -float("inf")
        category_decision = None
        c = hc.HumanCategories()
        for category in hc.get_human_object_recognition_categories():
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices)
            aggregated_value = self.aggregation_function(values)
            if aggregated_value > max_value:
                max_value = aggregated_value
                category_decision = category

        return category_decision
