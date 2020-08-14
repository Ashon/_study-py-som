
import numpy as np
import som_util

WEIGHT_VALUE_RANGE = som_util.ValueRange(min=0, max=1)


class FeatureMap(object):
    def __init__(self, width=0, height=0, dimension=0, randomize=False):
        if randomize:
            filler = np.random.rand
        else:
            filler = np.zeros

        self.map = filler(
            width * height * dimension
        ).reshape(width, height, dimension)

        self._width = width
        self._height = height
        self.dimension = dimension

    def get_bmu_coord(self, feature_vector):
        ''' returns best matching unit's coord '''

        errors = np.subtract(self.map, feature_vector)
        squared_errors = np.multiply(errors, errors)
        sum_squared_errors = np.sum(squared_errors, axis=2)
        min_error = np.amin(sum_squared_errors)
        min_error_address = np.where(sum_squared_errors == min_error)

        return [min_error_address[0][0], min_error_address[1][0]]

    def get_bmu(self, feature_vector):
        ''' returns bmu '''
        bmu_coord = self.get_bmu_coord(feature_vector)

        return self.map[bmu_coord]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


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
        bmu_coord = np.array(self.get_bmu_coord(feature_vector))
        gain = self._gain * (1 - self.get_progress())
        squared_gain = gain * gain

        coord_matrix = np.array([
            [x, y] for x in range(self._width) for y in range(self._height)
        ]).reshape(self._width, self._height, 2)

        distance_matrix = np.subtract(coord_matrix, bmu_coord)

        squared_dist_matrix = np.multiply(
            distance_matrix, distance_matrix).sum(axis=2)

        activation_map = np.multiply(
            np.exp(
                np.divide(-squared_dist_matrix, squared_gain)),
            self._learning_rate)

        feature_error_map = np.add(-self.map, feature_vector)

        for x, error_col, activate_col in zip(
                range(self._width), feature_error_map, activation_map):

            for y, feature_error, activate in zip(
                    range(self._height), error_col, activate_col):

                if activate >= self._learn_threshold:
                    bonus_weight = np.multiply(
                        feature_error, activate * self._learning_rate)
                    self.map[x][y] = np.clip(
                        np.add(self.map[x][y], bonus_weight), a_min=0, a_max=1)

    def train_feature_map(self, feature_map):
        for sample_unit in feature_map.map:
            self.train_feature_vector(sample_unit)
