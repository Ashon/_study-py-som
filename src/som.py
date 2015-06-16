
import numpy as np
import math

import som_util
import point_2d


WEIGHT_VALUE_RANGE = som_util.ValueRange(min=0, max=1)

class FeatureVector(point_2d.Point2D):

    def __init__(self, x=0, y=0, dimension=0, randomize=False):
        super(FeatureVector, self).__init__(x, y)

        if randomize:
            self.weights = np.array([som_util.get_random(
                min_value=WEIGHT_VALUE_RANGE.min,
                max_value=WEIGHT_VALUE_RANGE.max) for _ in range(dimension)])
        else:
            self.weights = np.array([0.0] * dimension)
        self.get_dimension = self.weights.__len__


class FeatureMap(object):

    def __init__(self, width=0, height=0, dimension=0, randomize=False):

        self.units = [
            FeatureVector(x=x, y=y, randomize=randomize, dimension=dimension)
            for x in range(width) for y in range(height)
        ]
        self.get_scale = self.units.__len__

        self._dimension = dimension

    def get_bmu(self, feature_vector):
        ''' returns best matching unit '''
        return self.units[self.get_bmu_index(feature_vector)]

    def get_bmu_index(self, feature_vector):
        ''' returns best matching unit's index '''
        feature_errors = [som_util.get_squared_error(unit, feature_vector) for unit in self.units]
        return feature_errors.index(min(feature_errors))


class Som(FeatureMap):

    def __init__(self, width=0, height=0, dimension=0, randomize=False,
                 threshold=0.5, learning_rate=0.05, max_iteration=5, gain=2):

        super(Som, self).__init__(
            width=width, height=height,
            dimension=dimension, randomize=randomize)

        self._threshold = som_util.clamp(threshold, 0, 1)
        self._learning_rate = som_util.clamp(learning_rate, 0, 1)
        self._learn_threshold = self._threshold * self._learning_rate

        self._gain = gain
        self._iteration = 0
        self._max_iteration_count = int(max_iteration)

        self._dimension = dimension

    def _set_learn_threshold(self):
        self._learn_threshold = self._threshold * self._learning_rate

    def set_learning_rate(self, learning_rate):
        self._learning_rate = som_util.clamp(learning_rate, 0, 1)
        self._set_learn_threshold()

    def get_learning_rate(self):
        return self._learning_rate

    def set_threshold(self, threshold):
        self._threshold = som_util.clamp(threshold, 0, 1)
        self._set_learn_threshold()

    def get_threshold(self):
        return self._threshold

    def get_learn_threshold(self):
        return self._learn_threshold

    def set_iteration_count(self, iteration_count):
        self._max_iteration_count = iteration_count

    def get_iteration_count(self):
        return self._max_iteration_count

    def do_progress(self):
        self._iteration += 1

    def get_progress(self):
        return float(self._iteration) / self._max_iteration_count

    def train_feature_vector(self, feature_vector):
        bmu = self.get_bmu(feature_vector)
        gain = self._gain * (1 - self.get_progress())
        gain = gain * gain

        for unit in self.units:
            activate = math.exp(-bmu.get_squared_distance(unit) / gain) * self._learning_rate
            if activate > self._learn_threshold:
                np.add(unit.weights, np.multiply(
                        np.subtract(feature_vector.weights, unit.weights), activate
                    )
                )

    def train_feature_map(self, feature_map):
        for sample_unit in feature_map.units:
            self.train_feature_vector(sample_unit)
