

import numpy as np
import math

import som_util
import point_2d


WEIGHT_VALUE_RANGE = som_util.ValueRange(min=0, max=1)

class FeatureVector(point_2d.Point2D):

    def __init__(self, x=0, y=0, dimension=0, randomize=False):
        super(FeatureVector, self).__init__(x, y)

        if randomize:
            fill_method = np.random.rand
        else:
            fill_method = np.zeros

        self.weights = fill_method(dimension)
        self.dimension = dimension

class FeatureMap(object):

    def __init__(self, width=0, height=0, dimension=0, randomize=False):

        if randomize:
            fill_method = np.random.rand
        else:
            fill_method = np.zeros

        self.map = fill_method(width * height * dimension).reshape(width, height, dimension)

        self._width = width
        self._height = height
        self.dimension = dimension

    def get_bmu_coord(self, feature_vector):
        ''' returns best matching unit's coord '''
        error_list = np.array([
            [x, y, som_util.get_squared_error(self.map[x][y], feature_vector)]
            for y in range(self._height) for x in range(self._width)
        ])

        minimum_error = np.max(np.min(error_list, axis=0))
        min_err_item = error_list[np.where(error_list == minimum_error)[0]]

        bmu_coord = np.array([min_err_item[0][0], min_err_item[0][1]])
        return bmu_coord

    def get_bmu(self, feature_vector):
        ''' returns bmu '''
        bmu_coord = self.get_bmu_coord(feature_vector)

        return self.map[bmu_coord]


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
        bmu_coord = self.get_bmu_coord(feature_vector)

        gain = self._gain * (1 - self.get_progress())
        gain = gain * gain

        coord_matrix = np.array([
            [x, y] for y in range(self._height) for x in range(self._width)
        ]).reshape(self._height, self._width, 2)

        distance_matrix = np.subtract(coord_matrix, bmu_coord)
        squared_dist_matrix = np.multiply(distance_matrix, -distance_matrix).sum(axis=2)

        activation_matrix = np.multiply(
            np.exp(
                np.divide(squared_dist_matrix, gain)
            ),
            self._learning_rate
        )

        self.map = np.add(
            self.map,
            np.multiply(
                -np.subtract(self.map, feature_vector),
                activation_matrix
            )
        )

    def train_feature_map(self, feature_map):
        for sample_unit in feature_map.map:
            self.train_feature_vector(sample_unit)
